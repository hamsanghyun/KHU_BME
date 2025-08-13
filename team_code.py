import joblib
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from helper_code import *
from scipy.signal import butter, filtfilt, find_peaks, spectrogram, peak_widths
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

    def bandpass(data, low=0.5, high=40):
        nyq = 0.5 * fs
        b, a = butter(2, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, data - np.mean(data))

    preprocessed = {name: bandpass(lead(name)) for name in ['V1','V2','V3','V4','V6','II','I']}
    v1, v2, v3, v4, v6, ii, i_lead = [preprocessed[name] for name in ['V1','V2','V3','V4','V6','II','I']]

    def pan_tompkins(ecg):
        d = np.diff(ecg)
        s = d ** 2
        i = np.convolve(s, np.ones(int(0.15*fs))/int(0.15*fs), mode='same')
        threshold = 0.5 * np.max(i)
        peaks, _ = find_peaks(i, height=threshold, distance=int(0.2*fs))
        return peaks

    peaks = pan_tompkins(v2)
    rr = np.diff(peaks) / fs if len(peaks) > 1 else []
    hr = 60 / np.nanmean(rr) if len(rr) > 0 else 0
    rr_std = np.nanstd(rr) if len(rr) > 1 else 0
    rr_skew = skew(rr) if len(rr) > 2 else 0
    rr_kurt = kurtosis(rr) if len(rr) > 3 else 0

    qrs_dur, qt, tamp, q_dur, qrs_area = 0, 0, 0, 0, 0
    st_duration, st_slope, pr_interval, t_symmetry = 0, 0, 0, 0

    for p in peaks:
        right = min(p + int(0.4 * fs), len(v3))
        seg = v3[p:right]
        tpk, _ = find_peaks(seg, height=np.percentile(seg, 85))
        qt += (tpk[-1]/fs)*1000 if len(tpk) else 0

        seg2 = v4[p:right]
        base = np.median(v4[p-50:p]) if p >= 50 else 0
        tpk2, _ = find_peaks(seg2, height=np.percentile(seg2, 85))
        if len(tpk2):
            tamp += seg2[tpk2[-1]] - base
            half = len(tpk2) // 2
            left_amp = np.mean(seg2[tpk2[:half]]) if half > 0 else 0
            right_amp = np.mean(seg2[tpk2[half:]]) if half > 0 else 0
            t_symmetry += abs(left_amp - right_amp)

        seg_q = ii[max(p - int(0.25 * fs), 0):p]
        neg = np.where(seg_q < 0)[0]
        q_dur += (neg[-1] - neg[0]) / fs * 1000 if len(neg) > 1 else 0

        qrs_area += np.abs(v2[p]) if p < len(v2) else 0

        st_seg = v2[p + int(0.04*fs): p + int(0.08*fs)]
        st_duration += len(st_seg) / fs * 1000 if len(st_seg) else 0
        if len(st_seg) > 1:
            st_slope += (st_seg[-1] - st_seg[0]) / (len(st_seg) / fs)

        pr_start = max(p - int(0.2*fs), 0)
        pr_peak, _ = find_peaks(ii[pr_start:p], distance=5)
        if len(pr_peak):
            pr_interval += (p - (pr_start + pr_peak[-1])) / fs * 1000

    n_peaks = len(peaks) if len(peaks) else 1
    qt /= n_peaks; tamp /= n_peaks; q_dur /= n_peaks; qrs_area /= n_peaks
    st_duration /= n_peaks; st_slope /= n_peaks; pr_interval /= n_peaks; t_symmetry /= n_peaks

    qrs_dur = np.mean(peak_widths(v2, peaks, rel_height=0.5)[0])/fs*1000 if len(peaks) > 1 else 0
    qrs_dur = qrs_dur if np.isfinite(qrs_dur) else 0

    zcr = np.sum(np.diff(np.sign(v2)) != 0) / len(v2)

    def get_spectral(x):
        seg_len = min(len(x), 256)
        overlap = min(128, seg_len // 2)
        f, _, Sxx = spectrogram(x, fs=fs, nperseg=seg_len, noverlap=overlap)
        mask = f <= 40
        f, Sxx = f[mask], Sxx[mask]
        power = np.nan_to_num(np.mean(Sxx, axis=1))
        total = np.sum(power)
        lf = np.sum(power[(f >= 0) & (f < 15)])
        hf = np.sum(power[(f >= 15)])
        centroid = np.sum(f * power) / total if total else 0
        flatness = entropy(power / total) if total else 0
        return lf / hf if hf else 0, centroid, flatness

    def wavelet_entropy_vector(x):
        coeffs = pywt.wavedec(x, 'db4', level=4)
        energy = np.array([np.sum(c ** 2) for c in coeffs])
        norm_energy = energy / np.sum(energy) if np.sum(energy) > 0 else np.ones_like(energy)/len(energy)
        return norm_energy

    def cwt_features(x):
        cwt_matrix, _ = pywt.cwt(x, np.arange(1, 31), 'morl', sampling_period=1/fs)
        power = np.abs(cwt_matrix) ** 2
        return np.sum(power), np.mean(power), np.max(power)

    rsr_v1 = np.sum(np.diff(np.sign(np.diff(v1))) != 0)
    rsr_v2 = np.sum(np.diff(np.sign(np.diff(v2))) != 0)
    corr_iv2 = np.corrcoef(i_lead, v2)[0, 1] if np.std(i_lead) > 0 and np.std(v2) > 0 else 0

    e2 = np.nansum(v2 ** 2)
    v2_var = np.nanvar(v2)

    lf1, c1, f1 = get_spectral(i_lead)
    lf2, c2, f2 = get_spectral(ii)
    lf6, c6, f6 = get_spectral(v6)

    we_v2_vec = wavelet_entropy_vector(v2)
    we_i_vec = wavelet_entropy_vector(i_lead)
    cwt_sum, cwt_mean, cwt_max = cwt_features(v2)

    centroids, flatnesses = [], []
    for i in range(signal.shape[1]):
        _, c, f = get_spectral(signal[:, i])
        centroids.append(c)
        flatnesses.append(f)

    # Derived ratios
    qt_ratio = qt / qrs_dur if qrs_dur else 0
    q_dur_ratio = q_dur / qrs_dur if qrs_dur else 0
    qrs_area_ratio = qrs_area / (np.abs(v1).sum() / len(v1)) if len(v1) else 0

    return np.nan_to_num(np.array([
        hr, qrs_dur,
        lf1, lf2, lf6,
        rsr_v1, rsr_v2,
        *we_v2_vec, *we_i_vec,
        qt, tamp, q_dur,
        qrs_area,
        e2, v2_var, zcr, corr_iv2,
        rr_std, rr_skew, rr_kurt,
        cwt_sum, cwt_mean, cwt_max,
        st_duration, st_slope,
        pr_interval, t_symmetry,
        qt_ratio, q_dur_ratio, qrs_area_ratio,
        *centroids, *flatnesses
    ], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
