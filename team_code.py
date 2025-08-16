import os
import warnings
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from helper_code import *
from scipy.signal import butter, filtfilt, find_peaks, peak_widths, welch
from scipy.stats import entropy, skew, kurtosis
import pywt

np.random.seed(42)


# ---------- Threshold search (binary metrics only) ----------
def optimize_threshold(probas, labels):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.02, 0.5, 100):
        f1 = f1_score(labels, (probas >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


# ---------- Model I/O ----------
def save_model(model_folder, model):
    joblib.dump(model, os.path.join(model_folder, 'model_ensemble.sav'), protocol=0)


def load_model(model_folder, verbose):
    return joblib.load(os.path.join(model_folder, 'model_ensemble.sav'))


# ---------- Training ----------
def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the Challenge data...')
    records = find_records(data_folder)
    if len(records) == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print('Extracting features and labels...')
    feats, labels = [], []
    for i, rec in enumerate(records):
        if verbose:
            width = len(str(len(records)))
            print(f'- {i+1:>{width}}/{len(records)}: {rec}...')
        path = os.path.join(data_folder, rec)
        x = extract_features(path)
        if np.all(np.isfinite(x)):
            feats.append(x)
            labels.append(load_label(path))

    X_raw = np.vstack(feats).astype(np.float32)
    y = np.asarray(labels, dtype=int)

    # Global scaler for final fit/inference
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_raw)

    # ----- OOF ensemble + calibration (no leakage: fold-wise scaler/SMOTE) -----
    if verbose:
        print('Building OOF predictions for calibration and ensemble weighting...')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(y), dtype=float)
    oof_rf = np.zeros(len(y), dtype=float)

    for tr, va in skf.split(X_raw, y):
        Xtr_raw, ytr = X_raw[tr], y[tr]
        Xva_raw = X_raw[va]

        scaler_cv = StandardScaler().fit(Xtr_raw)
        Xtr = scaler_cv.transform(Xtr_raw)
        Xva = scaler_cv.transform(Xva_raw)

        Xtr_res, ytr_res = SMOTE(random_state=42).fit_resample(Xtr, ytr)

        xgb_cv = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.035,
            subsample=0.8, colsample_bytree=0.8, max_delta_step=1,
            tree_method='hist', eval_metric='logloss',
            random_state=42, use_label_encoder=False, scale_pos_weight=1.0
        ).fit(Xtr_res, ytr_res)

        rf_cv = RandomForestClassifier(
            n_estimators=600, max_depth=None, n_jobs=-1,
            class_weight=None, random_state=42
        ).fit(Xtr_res, ytr_res)

        oof_xgb[va] = xgb_cv.predict_proba(Xva)[:, 1]
        oof_rf[va] = rf_cv.predict_proba(Xva)[:, 1]

    # Ensemble weight tuned on Challenge score using OOF probabilities
    if verbose:
        print('Searching ensemble weight on Challenge score...')
    best_w, best_s = 0.5, -1.0
    for w in np.linspace(0.1, 0.9, 17):
        s = compute_challenge_score(y, w * oof_xgb + (1 - w) * oof_rf)
        if s > best_s:
            best_w, best_s = w, s

    # Calibrator learned on OOF ensemble scores
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        calibrator = IsotonicRegression(out_of_bounds='clip').fit(
            best_w * oof_xgb + (1 - best_w) * oof_rf, y
        )

    # ----- Final training on full data (single imbalance strategy: SMOTE only) -----
    if verbose:
        print('Training final models on full data...')
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_all, y)

    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.035,
        subsample=0.8, colsample_bytree=0.8, max_delta_step=1,
        tree_method='hist', eval_metric='logloss',
        random_state=42, use_label_encoder=False, scale_pos_weight=1.0
    ).fit(X_res, y_res)

    rf = RandomForestClassifier(
        n_estimators=600, max_depth=None, n_jobs=-1,
        class_weight=None, random_state=42
    ).fit(X_res, y_res)

    # Threshold only for binary reporting; probabilities are calibrated
    p_train_raw = best_w * xgb.predict_proba(X_all)[:, 1] + (1 - best_w) * rf.predict_proba(X_all)[:, 1]
    p_train = calibrator.predict(p_train_raw)
    threshold = optimize_threshold(p_train, y)

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, {
        'xgb': xgb,
        'rf': rf,
        'scaler': scaler,
        'calibrator': calibrator,
        'w_ens': float(best_w),
        'threshold': float(threshold)
    })

    if verbose:
        print(f'Ensemble weight: {best_w:.3f} | Calibrated F1-optimal threshold: {threshold:.3f}')


# ---------- Inference ----------
def run_model(record, model, verbose):
    x = extract_features(record).reshape(1, -1)
    x = model['scaler'].transform(x)
    px = float(model['xgb'].predict_proba(x)[0][1])
    pr = float(model['rf'].predict_proba(x)[0][1])
    p_raw = model.get('w_ens', 0.5) * px + (1 - model.get('w_ens', 0.5)) * pr
    calibrator = model.get('calibrator', _Identity())
    p = float(calibrator.predict([p_raw])[0])
    return int(p >= model['threshold']), p


class _Identity:
    def predict(self, x):
        return np.asarray(x, dtype=float)


# ---------- Feature extraction ----------
def extract_features(record):
    header = load_header(record)
    signal, fields = load_signals(record)
    fs = fields['fs']

    leads = get_signal_names(header)
    idx = {name: i for i, name in enumerate(leads)}

    def get_lead(name):
        if name in idx:
            return signal[:, idx[name]].astype(np.float64, copy=False)
        return np.zeros(signal.shape[0], dtype=np.float64)

    def bp(x, low=0.5, high=40.0, order=2):
        nyq = 0.5 * fs
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        pad = 3 * max(len(a), len(b))
        if len(x) <= pad:
            return x.copy()
        return filtfilt(b, a, x)

    def qrs_candidates(x):
        f = bp(x)
        d = np.diff(f)
        if d.size == 0:
            return np.array([], dtype=int)
        s = d * d
        w = max(1, int(0.150 * fs))
        integ = np.convolve(s, np.ones(w) / w, mode='same')
        init = integ[:min(len(integ), int(2 * fs))]
        if init.size == 0:
            return np.array([], dtype=int)
        sig, noi = 0.25 * np.max(init), 0.5 * np.mean(init)
        k = 0.25
        peaks, last = [], -np.inf
        min_rr = int(0.20 * fs)
        for n, val in enumerate(integ):
            thr = noi + k * (sig - noi)
            if val > thr and (n - last) > min_rr:
                L, R = max(n - int(0.05 * fs), 0), min(n + int(0.05 * fs), len(f) - 1)
                if R > L and f[n] == np.max(f[L:R + 1]):
                    peaks.append(n); last = n
                    sig = 0.125 * val + 0.875 * sig
            else:
                noi = 0.125 * val + 0.875 * noi
            if len(peaks) > 2:
                rr_avg = np.mean(np.diff(peaks[-8:])) if len(peaks) >= 3 else np.diff(peaks[-2:])[0]
                if (n - last) > 1.66 * rr_avg:
                    thr *= 0.5
        return np.asarray(peaks, dtype=int)

    def merge(peak_lists, tol_ms=40):
        arrs = [p for p in peak_lists if p.size > 0]
        allp = np.sort(np.concatenate(arrs)) if arrs else np.array([], dtype=int)
        if allp.size == 0:
            return allp
        tol = int(tol_ms * fs / 1000.0)
        merged = [allp[0]]
        for p in allp[1:]:
            if p - merged[-1] <= tol:
                merged[-1] = int((merged[-1] + p) * 0.5)
            else:
                merged.append(p)
        return np.asarray(merged, dtype=int)

    def safe_pct(x, q, default=0.0):
        x = np.asarray(x)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return default
        return float(np.percentile(x, q))

    def spectral_feats(x):
        if len(x) < 64:
            return 0.0, 0.0, 0.0
        f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
        m = f <= 40.0
        f, Pxx = f[m], Pxx[m]
        tot = float(np.sum(Pxx))
        if tot <= 0:
            return 0.0, 0.0, 0.0
        lf = float(np.sum(Pxx[(f >= 0) & (f < 15)]))
        hf = float(np.sum(Pxx[(f >= 15) & (f <= 40)]))
        lf_hf = float(lf / hf) if hf > 0 else 0.0
        centroid = float(np.sum(f * Pxx) / tot)
        flat = float(entropy(Pxx / tot))
        return lf_hf, centroid, flat

    def wavelet_ent(x):
        x = np.asarray(x)
        if x.size < 32:
            return 0.0
        try:
            max_lvl = pywt.dwt_max_level(len(x), pywt.Wavelet('db4').dec_len)
            lvl = min(4, max(1, max_lvl))
            coeffs = pywt.wavedec(x, 'db4', level=lvl)
            e = np.array([np.sum(c ** 2) for c in coeffs], dtype=np.float64)
            tot = e.sum()
            if tot <= 0:
                return 0.0
            p = e / tot
            return float(entropy(p))
        except Exception:
            return 0.0

    def cwt_feats(x):
        x = np.asarray(x)
        if x.size < 64:
            return 0.0, 0.0, 0.0
        widths = np.arange(1, 16)
        try:
            cwt_matrix, _ = pywt.cwt(x, widths, 'morl', sampling_period=1 / fs)
            power = np.abs(cwt_matrix) ** 2
            return float(np.sum(power)), float(np.mean(power)), float(np.max(power))
        except Exception:
            return 0.0, 0.0, 0.0

    # Leads
    v1 = get_lead('V1'); v2 = get_lead('V2'); v3 = get_lead('V3')
    v4 = get_lead('V4'); v5 = get_lead('V5'); v6 = get_lead('V6')
    ii = get_lead('II'); i_lead = get_lead('I')

    # Multi-lead R-peak consensus
    peaks = merge([qrs_candidates(ii), qrs_candidates(v2), qrs_candidates(v5)], tol_ms=40)

    # RR statistics
    rr = np.diff(peaks) / fs if peaks.size > 1 else np.array([])
    hr = float(60.0 / np.nanmean(rr)) if rr.size > 0 and np.nanmean(rr) > 0 else 0.0
    rr_std = float(np.nanstd(rr)) if rr.size > 1 else 0.0
    rr_sk = float(skew(rr)) if rr.size > 2 else 0.0
    rr_ku = float(kurtosis(rr)) if rr.size > 3 else 0.0

    # QRS duration (robust, median across beats)
    def qrs_duration_ms(x, r_peaks):
        if r_peaks.size < 2:
            return 0.0
        xb = bp(x, low=5.0, high=25.0, order=2)
        widths_ms = []
        win = int(0.08 * fs)
        for r in r_peaks:
            L, R = max(r - win, 0), min(r + win, len(xb) - 1)
            seg = xb[L:R]
            if seg.size < 3:
                continue
            d = np.abs(np.diff(seg))
            thr = np.percentile(d, 75)
            idx = np.where(d > thr)[0]
            if idx.size > 0:
                left = L + idx[0]
                right = L + idx[-1]
                widths_ms.append((right - left) / fs * 1000.0)
        return float(np.median(widths_ms)) if widths_ms else 0.0

    qrs_dur = qrs_duration_ms(v2 if np.any(v2) else ii, peaks)

    # QT, T amplitude, symmetry, Q duration
    qt, tamp, q_dur, t_sym = 0.0, 0.0, 0.0, 0.0
    n_beats = max(1, peaks.size)
    for p in peaks:
        right = min(p + int(0.40 * fs), len(v3))
        if right <= p:
            continue
        seg = v3[p:right]
        t_pk, _ = find_peaks(seg, height=safe_pct(seg, 85, default=(np.max(seg) * 0.85 if seg.size else 0.0)))
        if t_pk.size:
            qt += (t_pk[-1] / fs) * 1000.0

        seg2 = v4[p:right]
        base = float(np.median(v4[max(p - 50, 0):p])) if p > 0 else 0.0
        t_pk2, _ = find_peaks(seg2, height=safe_pct(seg2, 85, default=(np.max(seg2) * 0.85 if seg2.size else 0.0)))
        if t_pk2.size:
            tamp += float(seg2[t_pk2[-1]] - base)
            half = t_pk2.size // 2
            left_amp = float(np.mean(seg2[t_pk2[:half]])) if half > 0 else 0.0
            right_amp = float(np.mean(seg2[t_pk2[half:]])) if half > 0 else 0.0
            t_sym += abs(left_amp - right_amp)

        seg_q = ii[max(p - int(0.25 * fs), 0):p]
        neg = np.where(seg_q < 0)[0]
        if neg.size > 1:
            q_dur += (neg[-1] - neg[0]) / fs * 1000.0

    qt /= n_beats
    tamp /= n_beats
    q_dur /= n_beats
    t_sym /= n_beats

    base_sig = v2 if np.any(v2) else ii
    zcr = float(np.sum(np.diff(np.sign(base_sig)) != 0) / len(base_sig)) if len(base_sig) > 0 else 0.0

    # RSR-like simple counts
    def rsr_simple(xsig, r_peaks):
        if r_peaks.size == 0:
            return 0
        f = bp(xsig, low=5, high=15, order=2)
        d = np.diff(f)
        cnt = 0
        win = int(0.05 * fs)
        for r in r_peaks:
            L, R = max(r - win, 0), min(r + win, len(d))
            seg = d[L:R]
            if np.where(np.diff(np.sign(seg)))[0].size >= 3:
                cnt += 1
        return int(cnt)

    rsr_v1 = rsr_simple(v1, peaks)
    rsr_v2 = rsr_simple(v2, peaks)

    # Spectral/Wavelet features
    lf1, c1, f1 = spectral_feats(i_lead)
    lf2, c2, f2 = spectral_feats(ii)
    lf6, c6, f6 = spectral_feats(v6)

    we_v2 = wavelet_ent(base_sig)
    we_i = wavelet_ent(i_lead)

    cwt_sum, cwt_mean, cwt_max = cwt_feats(base_sig)

    # Global per-channel summaries
    centroids, flatnesses = [], []
    for ch in range(signal.shape[1]):
        _, c, fl = spectral_feats(signal[:, ch])
        centroids.append(c)
        flatnesses.append(fl)

    e2 = float(np.nansum(base_sig ** 2))
    v2_var = float(np.nanvar(base_sig))

    feats = np.array([
        hr, qrs_dur,
        lf1, lf2, lf6,
        rsr_v1, rsr_v2,
        we_v2, we_i,
        qt, tamp, q_dur,
        e2, v2_var, zcr,
        rr_std, rr_sk, rr_ku,
        cwt_sum, cwt_mean, cwt_max,
        *centroids, *flatnesses,
        t_sym
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
