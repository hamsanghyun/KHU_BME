import joblib
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from helper_code import *
from scipy.signal import butter, lfilter, find_peaks, spectrogram, peak_widths
from scipy.stats import entropy, skew, kurtosis
import pywt
import warnings

def optimize_threshold(probas, labels):
    best_thresh, best_score = 0.5, 0
    thresholds = np.linspace(0.01, 0.5, 200)
    for t in thresholds:
        outputs = probas * (probas >= t)
        score = compute_challenge_score(labels, outputs)
        if score > best_score:
            best_thresh, best_score = t, score
    return best_thresh

def save_model(model_folder, model):
    joblib.dump(model, os.path.join(model_folder, 'model_ensemble.sav'), protocol=0)

def load_model(model_folder, verbose):
    return joblib.load(os.path.join(model_folder, 'model_ensemble.sav'))

def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the Challenge data...')
    records = find_records(data_folder)
    if len(records) == 0:
        raise FileNotFoundError('No data found.')

    features_list, labels = [], []
    for i, rec in enumerate(records):
        if verbose:
            print(f'- {i+1}/{len(records)}: {rec}...')
        record_path = os.path.join(data_folder, rec)
        feat = extract_features(record_path)
        if np.all(np.isfinite(feat)):
            features_list.append(feat)
            labels.append(load_label(record_path))

    features = np.vstack(features_list)
    labels = np.array(labels)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    smote = SMOTE()
    features_res, labels_res = smote.fit_resample(features, labels)

    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1

    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        use_label_encoder=False, eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        max_delta_step=1, subsample=0.8, colsample_bytree=0.8
    )
    xgb.fit(features_res, labels_res)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        class_weight='balanced', n_jobs=-1,
        random_state=42
    )
    rf.fit(features_res, labels_res)

    probas_xgb = xgb.predict_proba(features)[:, 1]
    probas_rf = rf.predict_proba(features)[:, 1]
    probas_ensemble = (probas_xgb + probas_rf) / 2

    threshold = optimize_threshold(probas_ensemble, labels)

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, {
        'xgb': xgb,
        'rf': rf,
        'threshold': threshold,
        'scaler': scaler
    })

    if verbose:
        print(f'Training done. Best threshold: {threshold:.3f}')

def run_model(record, model, verbose):
    features = extract_features(record).reshape(1, -1)
    features = model['scaler'].transform(features)
    proba_xgb = model['xgb'].predict_proba(features)[0][1]
    proba_rf = model['rf'].predict_proba(features)[0][1]
    proba = (proba_xgb + proba_rf) / 2
    return int(proba >= model['threshold']), proba

def extract_features(record):
    header = load_header(record)
    signal, fields = load_signals(record)
    fs = fields['fs']
    leads = get_signal_names(header)
    lead_map = {name: i for i, name in enumerate(leads)}
    def lead(name): return signal[:, lead_map.get(name, 0)]

    def bandpass_filter(data, lowcut=0.5, highcut=40.0):
        nyq = 0.5 * fs
        b, a = butter(1, [lowcut / nyq, highcut / nyq], btype='band')
        return lfilter(b, a, data)

    def pan_tompkins_qrs(ecg):
        f = bandpass_filter(ecg)
        d = np.diff(f)
        if d.size == 0:
            return np.array([], dtype=int)
        s = d ** 2
        w = max(1, int(0.150 * fs))
        i = np.convolve(s, np.ones(w)/w, mode='same')
        init_len = min(len(i), int(2 * fs))
        initial_seg = i[:init_len] if init_len > 0 else i
        if initial_seg.size == 0:
            return np.array([], dtype=int)
        SignalLevel = 0.25 * np.max(initial_seg)
        NoiseLevel = 0.5 * np.mean(initial_seg)
        Threshold_coefficient = 0.25

        peaks = []
        last_peak = -np.inf
        min_rr = int(0.20 * fs)

        for n, val in enumerate(i):
            threshold = NoiseLevel + Threshold_coefficient * (SignalLevel - NoiseLevel)
            if val > threshold and (n - last_peak) > min_rr:
                left = max(n - int(0.05 * fs), 0)
                right = min(n + int(0.05 * fs), len(f)-1)
                if right > left and f[n] == np.max(f[left:right+1]):
                    peaks.append(n)
                    last_peak = n
                    SignalLevel = 0.125 * val + 0.875 * SignalLevel
            else:
                NoiseLevel = 0.125 * val + 0.875 * NoiseLevel

            if len(peaks) > 1:
                rr_avg = np.mean(np.diff(peaks[-8:])) if len(peaks) >= 3 else np.diff(peaks[-2:])[0]
                if (n - last_peak) > 1.66 * rr_avg:
                    threshold *= 0.5

        return np.asarray(peaks, dtype=int)

    def safe_percentile(x, q, default=0.0):
        x = np.asarray(x)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return default
        return float(np.percentile(x, q))

    def get_spectral(x):
        segment_length = min(len(x), 256)
        if segment_length <= 1:
            return 0.0, 0.0, 0.0
        f, _, Sxx = spectrogram(x, fs=fs, nperseg=segment_length, noverlap=segment_length//2)
        mask = f <= 40
        f, Sxx = f[mask], Sxx[mask]
        if Sxx.size == 0:
            return 0.0, 0.0, 0.0
        power = np.nan_to_num(np.mean(Sxx, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
        total = np.sum(power)
        if total <= 0:
            return 0.0, 0.0, 0.0
        lf = np.sum(power[(f >= 0) & (f < 15)])
        hf = np.sum(power[(f >= 15) & (f <= 40)])
        lf_hf = float(lf / hf) if hf > 0 else 0.0
        centroid = float(np.sum(f * power) / total)
        flatness = float(entropy(power / total))
        return lf_hf, centroid, flatness

    def wavelet_entropy(x):
        x = np.asarray(x)
        if x.size < 8:
            return 0.0
        max_lvl = pywt.dwt_max_level(len(x), pywt.Wavelet('db4').dec_len)
        lvl = min(4, max(1, max_lvl))
        coeffs = pywt.wavedec(x, 'db4', level=lvl)
        energy = np.array([np.sum(c**2) for c in coeffs], dtype=np.float64)
        total = energy.sum()
        if total <= 0:
            return 0.0
        p = energy / total
        return float(entropy(p))

    def cwt_features(x):
        x = np.asarray(x)
        if x.size < 8:
            return 0.0, 0.0, 0.0
        widths = np.arange(1, 31)
        cwt_matrix, _ = pywt.cwt(x, widths, 'morl', sampling_period=1/fs)
        power = np.abs(cwt_matrix) ** 2
        return float(np.sum(power)), float(np.mean(power)), float(np.max(power))

    v1, v2, v3, v4, v6, ii, i_lead = [lead(l) for l in ['V1','V2','V3','V4','V6','II','I']]

    peaks = pan_tompkins_qrs(v2)
    rr = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])
    heart_rate = float(60 / np.nanmean(rr)) if rr.size > 0 and np.nanmean(rr) > 0 else 0.0
    rr_std = float(np.nanstd(rr)) if rr.size > 1 else 0.0
    rr_skew = float(skew(rr)) if rr.size > 2 else 0.0
    rr_kurt = float(kurtosis(rr)) if rr.size > 3 else 0.0

    if len(peaks) > 1:
        widths = peak_widths(v2, peaks, rel_height=0.5)[0]
        qrs_dur = float(np.nanmean(widths) / fs * 1000.0) if widths.size else 0.0
    else:
        qrs_dur = 0.0
    if not np.isfinite(qrs_dur):
        qrs_dur = 0.0

    qt, tamp, t_symmetry = 0.0, 0.0, 0.0
    q_dur = 0.0
    n_peaks = max(1, len(peaks))

    for p in peaks:
        right = min(p + int(0.4 * fs), len(v3))
        if right <= p:
            continue

        # QT from V3
        seg = v3[p:right]
        thr = safe_percentile(seg, 85, default=(np.max(seg) * 0.85 if seg.size else 0.0))
        t_pk, _ = find_peaks(seg, height=thr)
        if t_pk.size:
            qt += (t_pk[-1] / fs) * 1000.0

        # T amplitude & symmetry from V4
        seg2 = v4[p:right]
        base_slice = v4[max(p - 50, 0):p]
        base = float(np.median(base_slice)) if base_slice.size else 0.0
        thr2 = safe_percentile(seg2, 85, default=(np.max(seg2) * 0.85 if seg2.size else 0.0))
        t_pk2, _ = find_peaks(seg2, height=thr2)
        if t_pk2.size:
            tamp += float(seg2[t_pk2[-1]] - base)
            half = t_pk2.size // 2
            left_amp = float(np.mean(seg2[t_pk2[:half]])) if half > 0 else 0.0
            right_amp = float(np.mean(seg2[t_pk2[half:]])) if half > 0 else 0.0
            t_symmetry += abs(left_amp - right_amp)

        # Q-duration from lead II
        seg_q = ii[max(p - int(0.25 * fs), 0):p]
        neg = np.where(seg_q < 0)[0]
        if neg.size > 1:
            q_dur += (neg[-1] - neg[0]) / fs * 1000.0

    qt /= n_peaks
    tamp /= n_peaks
    q_dur /= n_peaks
    t_symmetry /= n_peaks

    zcr = float(np.sum(np.diff(np.sign(v2)) != 0) / len(v2)) if len(v2) > 0 else 0.0

    # RSR count
    def detect_rsr_ensemble(xsig, r_peaks, fs):
        def _bp(x):
            b, a = butter(2, [5/(fs/2), 15/(fs/2)], btype='band')
            return lfilter(b, a, x)
        if r_peaks.size == 0:
            return 0
        f = _bp(xsig)
        d = np.diff(f)
        count1 = 0
        for p in r_peaks:
            left, right = max(p - int(0.05 * fs), 0), min(p + int(0.05 * fs), len(d))
            seg = d[left:right]
            zc = np.where(np.diff(np.sign(seg)))[0]
            if zc.size >= 3:
                a, b, c = f[left:right][zc[:3]]
                if a > b and c > b and abs(a - c) < 0.1 * max(abs(a), abs(c)) and (max([a,b,c]) - min([a,b,c])) > 0.05:
                    count1 += 1
        count2 = 0
        for i in range(1, len(r_peaks)):
            p1, p2 = r_peaks[i - 1], r_peaks[i]
            rr = (p2 - p1) / fs
            if rr < 0.12 and (p2 - p1) > 2:
                seg = xsig[p1:p2]
                slope = np.diff(seg)
                if slope.size > 2:
                    sc = np.diff(np.sign(slope))
                    z = np.where(sc != 0)[0]
                    if any(slope[zi] < 0 and slope[zi + 1] > 0 for zi in z[:-1]):
                        count2 += 1
        return max(count1, count2)

    rsr_v1 = detect_rsr_ensemble(v1, peaks, fs)
    rsr_v2 = detect_rsr_ensemble(v2, peaks, fs)

    e2 = float(np.nansum(v2 ** 2))
    v2_var = float(np.nanvar(v2))

    lf1, c1, f1 = get_spectral(i_lead)
    lf2, c2, f2 = get_spectral(ii)
    lf6, c6, f6 = get_spectral(v6)

    we_v2 = wavelet_entropy(v2)
    we_i = wavelet_entropy(i_lead)
    cwt_sum, cwt_mean, cwt_max = cwt_features(v2)

    centroids, flatnesses = [], []
    for ch in range(signal.shape[1]):
        _, c, f = get_spectral(signal[:, ch])
        centroids.append(c)
        flatnesses.append(f)

    return np.nan_to_num(np.array([
        heart_rate, qrs_dur,
        lf1, lf2, lf6,
        rsr_v1, rsr_v2,
        we_v2, we_i,
        qt, tamp, q_dur,
        e2, v2_var, zcr,
        rr_std, rr_skew, rr_kurt,
        cwt_sum, cwt_mean, cwt_max,
        *centroids, *flatnesses,
        t_symmetry
    ], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

