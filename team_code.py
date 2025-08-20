# team_code.py
# 저장/로드 방식:
# - 파일명 고정: model_rf+xgb+lr.sav
# - MANIFEST.json, LATEST.txt 관리
# - legacy(model_xgb.sav) 역호환
# 모델 고정: rf+xgb+lr
# 임계값: OOF 기반 F1 최적(threshold_f1), Challenge 상위5% 컷(threshold_challenge)
# 교차검증: StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# payload에 fold별 점수/배열 저장(scores, oof_label, oof_proba, fold_ids)

import os
import json
import joblib
import numpy as np
from typing import Dict, Tuple, List

from helper_code import (
    load_header, load_signals, get_signal_names, find_records, load_label,
    compute_auc, compute_accuracy, compute_f_measure, compute_challenge_score
)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from scipy.signal import butter, filtfilt, lfilter, welch, spectrogram, peak_widths
from scipy.stats import entropy, skew, kurtosis

# =========================================================
# 전역
# =========================================================
RNG = 42
np.random.seed(RNG)

# =========================================================
# 저장/로드 유틸
# =========================================================
def _model_path(model_folder: str) -> str:
    return os.path.join(model_folder, "model_rf+xgb+lr.sav")

def _manifest_path(model_folder: str) -> str:
    return os.path.join(model_folder, "MANIFEST.json")

def _latest_path(model_folder: str) -> str:
    return os.path.join(model_folder, "LATEST.txt")

def save_payload(model_folder: str, payload: dict):
    os.makedirs(model_folder, exist_ok=True)
    path = _model_path(model_folder)
    joblib.dump(payload, path, protocol=0)

    meta = {
        "kind": "rf+xgb+lr",
        "members": payload.get("members"),
        "weights": payload.get("weights"),
        "feature_dim": payload.get("feature_dim"),
        "threshold_challenge": payload.get("threshold_challenge"),
        "threshold_f1": payload.get("threshold_f1"),
        "scaler": "StandardScaler",
        "file": os.path.basename(path),
    }
    # MANIFEST append/update
    manifest_fp = _manifest_path(model_folder)
    try:
        if os.path.exists(manifest_fp):
            with open(manifest_fp, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if not isinstance(manifest, list):
                manifest = [manifest]
        else:
            manifest = []
    except Exception:
        manifest = []
    manifest.append(meta)
    with open(manifest_fp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # LATEST.txt 업데이트
    with open(_latest_path(model_folder), "w", encoding="utf-8") as f:
        f.write(os.path.basename(path))

def load_payload(model_folder: str, verbose: bool):
    # 고정 파일 우선
    fixed = _model_path(model_folder)
    if os.path.exists(fixed):
        if verbose:
            print(f"Loading model file: {os.path.basename(fixed)}")
        return joblib.load(fixed)

    # 최신 포인터
    latest_fp = _latest_path(model_folder)
    if os.path.exists(latest_fp):
        with open(latest_fp, "r", encoding="utf-8") as f:
            latest_name = f.read().strip()
        latest_path = os.path.join(model_folder, latest_name)
        if os.path.exists(latest_path):
            if verbose:
                print(f"Loading latest model: {latest_name}")
            return joblib.load(latest_path)

    # legacy
    legacy = os.path.join(model_folder, "model_xgb.sav")
    if os.path.exists(legacy):
        if verbose:
            print("Loading legacy file: model_xgb.sav")
        return joblib.load(legacy)

    # fallback
    for fn in sorted(os.listdir(model_folder)):
        if fn.startswith("model_") and fn.endswith(".sav"):
            if verbose:
                print(f"Loading fallback: {fn}")
            return joblib.load(os.path.join(model_folder, fn))

    raise FileNotFoundError("No model file found.")

# 호환 래퍼 (평가 스크립트 호환용)
def save_model(model_folder, model, name='rf+xgb+lr'):
    save_payload(model_folder, model)

# =========================================================
# 임계값 최적화
# =========================================================
def optimize_threshold_f1(probas: np.ndarray, labels: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 197):
        preds = (probas >= t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t)

def optimize_threshold_top5(probas: np.ndarray) -> float:
    n = len(probas)
    if n == 0:
        return 1.0
    k = max(1, int(np.floor(0.05 * n)))
    thr = np.partition(probas, -k)[-k]
    return float(np.nextafter(thr, np.float64(np.inf)))

# =========================================================
# 신호 처리 유틸
# =========================================================
def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a

def bandpass_filter(x: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 40.0, order: int = 2):
    if x.size < 4:
        return x
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    try:
        return filtfilt(b, a, x, method="gust")
    except Exception:
        return lfilter(b, a, x)

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w, dtype=float) / w
    return np.convolve(x, k, mode="same")

# =========================================================
# R-피크 검출
# =========================================================
def pan_tompkins_like(ecg: np.ndarray, fs: float) -> np.ndarray:
    if ecg.size < int(0.5 * fs):
        return np.array([], dtype=int)
    f = bandpass_filter(ecg, fs, 5.0, 15.0, order=2)
    d = np.diff(f, prepend=f[0])
    s = d ** 2
    w = max(1, int(0.150 * fs))
    integ = moving_average(s, w)

    init = integ[: min(len(integ), 2 * int(fs))]
    if init.size == 0:
        return np.array([], dtype=int)

    signal_level = 0.25 * np.max(init)
    noise_level = 0.5 * np.mean(init)
    c = 0.25
    min_rr = int(0.20 * fs)

    peaks, last = [], -10**9
    for n, v in enumerate(integ):
        thr = noise_level + c * (signal_level - noise_level)
        if v > thr and (n - last) > min_rr:
            left = max(0, n - int(0.05 * fs))
            right = min(len(f), n + int(0.05 * fs))
            loc = left + int(np.argmax(f[left:right]))
            peaks.append(loc)
            last = loc
            signal_level = 0.125 * v + 0.875 * signal_level
        else:
            noise_level = 0.125 * v + 0.875 * noise_level
    return np.asarray(peaks, dtype=int)

def robust_rpeaks(multilead: Dict[str, np.ndarray], fs: float) -> np.ndarray:
    cand = [nm for nm in ["II", "V2", "V1", "V5"] if nm in multilead]
    if len(cand) < 2:
        alln = list(multilead.keys())
        alln.sort(key=lambda k: np.nanvar(multilead[k]), reverse=True)
        cand = alln[:2]
    peak_lists = []
    for nm in cand:
        p = pan_tompkins_like(multilead[nm], fs)
        if p.size:
            peak_lists.append(p)
    if not peak_lists:
        return np.array([], dtype=int)
    all_peaks = np.concatenate(peak_lists)
    all_peaks.sort()
    merged = []
    tol = int(0.06 * fs)
    grp = [all_peaks[0]]
    for idx in all_peaks[1:]:
        if idx - grp[-1] <= tol:
            grp.append(idx)
        else:
            merged.append(int(np.median(grp)))
            grp = [idx]
    merged.append(int(np.median(grp)))
    return np.asarray(merged, dtype=int)

# =========================================================
# 형태학/스펙트럼/HRV 특징
# =========================================================
def beat_seg(signal: np.ndarray, peak: int, fs: float, pre: float = 0.20, post: float = 0.40) -> Tuple[int, int]:
    l = max(0, peak - int(pre * fs))
    r = min(len(signal), peak + int(post * fs))
    return l, r

def qrs_width_ms(x: np.ndarray, peaks: np.ndarray, fs: float) -> float:
    if peaks.size < 1:
        return 0.0
    try:
        widths, _, _, _ = peak_widths(x, peaks, rel_height=0.5)
        return float(np.nanmean(widths) / fs * 1000.0)
    except Exception:
        return 0.0

def rsr_fragmentation_count(x: np.ndarray, peaks: np.ndarray, fs: float) -> int:
    if peaks.size == 0:
        return 0
    xf = bandpass_filter(x, fs, 5, 30, order=2)
    d = np.diff(xf, prepend=xf[0])
    cnt = 0
    win = int(0.05 * fs)
    for p in peaks:
        l = max(0, p - win)
        r = min(len(d), p + win)
        seg = d[l:r]
        zc = np.where(np.diff(np.sign(seg)) != 0)[0]
        if len(zc) >= 3:
            cnt += 1
    return int(cnt)

def st_deviation_mv(x: np.ndarray, peaks: np.ndarray, fs: float) -> float:
    if peaks.size == 0:
        return 0.0
    vals = []
    j_offset = int(0.06 * fs)
    pre_pq_l = int(0.20 * fs)
    pre_pq_r = int(0.08 * fs)
    for p in peaks:
        j = min(len(x) - 1, p + j_offset)
        a = max(0, p - pre_pq_l)
        b = max(0, p - pre_pq_r)
        base = np.median(x[a:b]) if b > a else 0.0
        vals.append(x[j] - base)
    return float(np.nanmean(vals))

def t_wave_amp(x: np.ndarray, peaks: np.ndarray, fs: float) -> float:
    if peaks.size == 0:
        return 0.0
    amps = []
    for p in peaks:
        l = p + int(0.20 * fs)
        r = min(len(x), p + int(0.40 * fs))
        if r > l:
            amps.append(np.max(x[l:r]) - np.median(x[max(0, p - int(0.1 * fs)):p]))
    return float(np.nanmean(amps) if amps else 0.0)

def rs_ratio(x: np.ndarray, peaks: np.ndarray, fs: float) -> float:
    if peaks.size == 0:
        return 0.0
    ratios = []
    for p in peaks:
        l, r = beat_seg(x, p, fs)
        seg = x[l:r]
        if seg.size < 3:
            continue
        rp = int(np.argmax(seg))
        s_min = np.min(seg[rp:]) if seg.size - rp > 1 else seg[rp]
        r_amp = seg[rp] - np.median(seg[: max(1, rp)])
        s_amp = np.median(seg[: max(1, rp)]) - s_min
        if s_amp != 0:
            ratios.append(r_amp / s_amp)
    return float(np.nanmean(ratios) if ratios else 0.0)

def psd_bandpowers(x: np.ndarray, fs: float, bands=((0.5, 5), (5, 15), (15, 40))) -> List[float]:
    if x.size < 8:
        return [0.0] * (len(bands) + 3)
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 256), noverlap=None)
    total = np.trapz(Pxx, f) if Pxx.size else 0.0
    feats = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        val = np.trapz(Pxx[m], f[m]) if np.any(m) else 0.0
        feats.append(val)
    if total > 0:
        centroid = float(np.trapz(f * Pxx, f) / total)
        pow_norm = Pxx / np.sum(Pxx)
        flatness = float(entropy(pow_norm))
        cdf = np.cumsum(pow_norm)
        f_lo = f[np.searchsorted(cdf, 0.05)]
        f_hi = f[min(len(f) - 1, np.searchsorted(cdf, 0.95))]
        bandwidth = float(f_hi - f_lo)
    else:
        centroid = 0.0
        flatness = 0.0
        bandwidth = 0.0
    feats += [centroid, bandwidth, flatness]
    return feats

def spec_summary(x: np.ndarray, fs: float) -> Tuple[float, float, float]:
    if x.size < 32:
        return 0.0, 0.0, 0.0
    nseg = min(len(x), 256)
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=nseg, noverlap=nseg//2)
    mask = f <= 40.0
    f = f[mask]
    P = np.nan_to_num(Sxx[mask].mean(axis=1))
    tot = P.sum()
    if tot <= 0:
        return 0.0, 0.0, 0.0
    centroid = float(np.sum(f * P) / tot)
    pnorm = P / tot
    flat = float(entropy(pnorm))
    rolloff = f[np.searchsorted(np.cumsum(pnorm), 0.85)] if np.any(pnorm) else 0.0
    return centroid, flat, float(rolloff)

def zero_cross_rate(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    return float(np.mean(np.diff(np.sign(x)) != 0))

def axis_from_I_II(lead_I: np.ndarray, lead_II: np.ndarray, peaks: np.ndarray, fs: float) -> Tuple[float, float, float]:
    if peaks.size == 0:
        return 0.0, 0.0, 0.0
    rI, rII = [], []
    for p in peaks:
        l, r = beat_seg(lead_I, p, fs)
        segI = lead_I[l:r]
        l2, r2 = beat_seg(lead_II, p, fs)
        segII = lead_II[l2:r2]
        if segI.size and segII.size:
            rI.append(np.max(segI) - np.median(segI[: max(1, len(segI)//4)]))
            rII.append(np.max(segII) - np.median(segII[: max(1, len(segII)//4)]))
    if not rI or not rII:
        return 0.0, 0.0, 0.0
    RI = float(np.nanmean(rI))
    RII = float(np.nanmean(rII))
    angle = float(np.degrees(np.arctan2(RII, RI)))
    return angle, float(np.sin(np.radians(angle))), float(np.cos(np.radians(angle)))

# =========================================================
# 메인 피처 함수(100+ 차원)
# =========================================================
def extract_features(record: str) -> np.ndarray:
    header = load_header(record)
    signal, fields = load_signals(record)
    fs = float(fields["fs"])

    lead_names = get_signal_names(header)
    lead_idx = {name: i for i, name in enumerate(lead_names)}

    def get(name: str) -> np.ndarray:
        if name in lead_idx:
            return signal[:, lead_idx[name]].astype(np.float32, copy=False)
        return signal[:, 0].astype(np.float32, copy=False)

    leads = {
        "I": get("I"), "II": get("II"),
        "V1": get("V1"), "V2": get("V2"), "V3": get("V3"),
        "V4": get("V4"), "V5": get("V5"), "V6": get("V6"),
    }
    for k in leads:
        leads[k] = bandpass_filter(leads[k], fs, 0.5, 40.0, order=2)

    peaks = robust_rpeaks(leads, fs)
    rr = np.diff(peaks) / fs if peaks.size >= 2 else np.array([], dtype=float)
    mean_rr = float(np.nanmean(rr)) if rr.size else 0.0
    heart_rate = float(60.0 / mean_rr) if mean_rr > 0 else 0.0
    sdnn = float(np.nanstd(rr)) if rr.size else 0.0
    diff_rr = np.diff(rr) if rr.size >= 2 else np.array([], dtype=float)
    rmssd = float(np.sqrt(np.nanmean(diff_rr**2))) if diff_rr.size else 0.0
    pnn50 = float(np.mean(np.abs(diff_rr) > 0.05)) if diff_rr.size else 0.0
    rr_sk = float(skew(rr)) if rr.size >= 3 else 0.0
    rr_ku = float(kurtosis(rr)) if rr.size >= 4 else 0.0

    qrs_v1 = qrs_width_ms(leads["V1"], peaks, fs)
    qrs_v2 = qrs_width_ms(leads["V2"], peaks, fs)
    rsr_v1 = rsr_fragmentation_count(leads["V1"], peaks, fs)
    rsr_v2 = rsr_fragmentation_count(leads["V2"], peaks, fs)
    st_v2 = st_deviation_mv(leads["V2"], peaks, fs)
    st_v5 = st_deviation_mv(leads["V5"], peaks, fs)
    t_v3 = t_wave_amp(leads["V3"], peaks, fs)
    t_v4 = t_wave_amp(leads["V4"], peaks, fs)
    rs_ratio_v1 = rs_ratio(leads["V1"], peaks, fs)
    rs_ratio_v2 = rs_ratio(leads["V2"], peaks, fs)

    axis_deg, axis_sin, axis_cos = axis_from_I_II(leads["I"], leads["II"], peaks, fs)

    # PSD 요약(8리드 × 6 = 48)
    psd_feats = []
    for nm in ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]:
        psd_feats += psd_bandpowers(leads[nm], fs)

    # 스펙트로그램 요약(II, V2, V6 × 3 = 9)
    spec_feats = []
    for nm in ["II", "V2", "V6"]:
        c, fl, ro = spec_summary(leads[nm], fs)
        spec_feats += [c, fl, ro]

    # ZCR(8)
    zcrs = [zero_cross_rate(leads[nm]) for nm in ["I","II","V1","V2","V3","V4","V5","V6"]]

    # 에너지(8) + 분산(8) + MAD(8)
    energies = [float(np.nansum(leads[nm] ** 2)) for nm in ["I","II","V1","V2","V3","V4","V5","V6"]]
    variances = [float(np.nanvar(leads[nm])) for nm in ["I","II","V1","V2","V3","V4","V5","V6"]]
    mads = [float(np.nanmedian(np.abs(leads[nm] - np.nanmedian(leads[nm])))) for nm in ["I","II","V1","V2","V3","V4","V5","V6"]]

    feats = np.array([
        # HRV(7)
        heart_rate, mean_rr, sdnn, rmssd, pnn50, rr_sk, rr_ku,
        # 형태학/단편화/ST/T/RS(10)
        qrs_v1, qrs_v2, rsr_v1, rsr_v2, rs_ratio_v1, rs_ratio_v2, st_v2, st_v5, t_v3, t_v4,
        # 전기축(3)
        axis_deg, axis_sin, axis_cos,
        # PSD(48)
        *psd_feats,
        # 스펙트로그램 요약(9)
        *spec_feats,
        # ZCR(8)
        *zcrs,
        # 에너지/분산/MAD(24)
        *energies, *variances, *mads,
    ], dtype=np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats

# =========================================================
# 모델 빌더(고정 3종)
# =========================================================
def build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=800,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RNG,
    )

def build_xgb(y: np.ndarray) -> XGBClassifier:
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    spw = neg / pos
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=0.6,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=0.5,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=spw,
        n_jobs=0,
        random_state=RNG,
        verbosity=0,
    )

def build_lr() -> LogisticRegression:
    return LogisticRegression(
        C=10.0,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=RNG,
    )

def smote_fit_resample(X, y, seed):
    if (y == 1).sum() < 2 or (y == 0).sum() < 2:
        return X, y
    try:
        sm = SMOTE(random_state=seed)
        return sm.fit_resample(X, y)
    except Exception:
        return X, y

# =========================================================
# 학습
# =========================================================
def train_model(data_folder: str, model_folder: str, verbose: bool):
    if verbose:
        print("Finding the Challenge data...")
    records = find_records(data_folder)
    if len(records) == 0:
        raise FileNotFoundError("No data were provided.")

    if verbose:
        print("Extracting features and labels...")
    feats, labels = [], []
    for i, rec in enumerate(records, 1):
        if verbose and (i % 200 == 0 or i == len(records)):
            print(f"- {i}/{len(records)}: {rec}")
        path = os.path.join(data_folder, rec)
        f = extract_features(path)
        if np.all(np.isfinite(f)):
            feats.append(f)
            labels.append(load_label(path))

    X = np.vstack(feats).astype(np.float32)
    y = np.asarray(labels, dtype=int)

    if verbose:
        print(f"Feature matrix: {X.shape}, Positives: {int(y.sum())}, Negatives: {int((y==0).sum())}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    members: List[str] = ["rf", "xgb", "lr"]

    # --------- OOF 확률 산출 ---------
    if verbose:
        print("Building OOF predictions for kind = rf+xgb+lr ...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RNG)

    oof = {m: np.zeros(len(y), dtype=float) for m in members}
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xs, y), 1):
        X_tr, y_tr = Xs[tr_idx], y[tr_idx]
        X_va = Xs[va_idx]

        X_tr_res, y_tr_res = smote_fit_resample(X_tr, y_tr, seed=RNG + fold)

        rf = build_rf()
        rf.fit(X_tr_res, y_tr_res)
        oof["rf"][va_idx] = rf.predict_proba(X_va)[:, 1]

        xgb = build_xgb(y_tr_res)
        xgb.fit(X_tr_res, y_tr_res)
        oof["xgb"][va_idx] = xgb.predict_proba(X_va)[:, 1]

        lr = build_lr()
        lr.fit(X_tr_res, y_tr_res)
        oof["lr"][va_idx] = lr.predict_proba(X_va)[:, 1]

        fold_ids[va_idx] = fold

        if verbose:
            f1_tmp = {m: f1_score(y[va_idx], (oof[m][va_idx] >= 0.5).astype(int)) for m in members}
            msg = "  Fold {}: ".format(fold) + " ".join([f"{m} F1@0.5={f1_tmp[m]:.3f}" for m in members])
            print(msg)

    # --------- 앙상블 가중(OOF AUPRC 기반) ---------
    weights = {m: 1.0 for m in members}
    for m in members:
        _, auprc_m = compute_auc(y, oof[m])
        weights[m] = max(auprc_m, 1e-6)
    s = sum(weights.values())
    for m in members:
        weights[m] /= s

    # OOF 앙상블 확률
    w = np.array([weights[m] for m in members], dtype=float)   # [3]
    oof_stack = np.vstack([oof[m] for m in members])           # [3, N]
    oof_ens = (w[:, None] * oof_stack).sum(axis=0)

    # 임계값 산출
    thr_f1 = optimize_threshold_f1(oof_ens, y)
    thr_chal = optimize_threshold_top5(oof_ens)
    if verbose:
        print(f"Optimized thresholds -> F1: {thr_f1:.4f}, Top5%: {thr_chal:.6f}")

    # --------- fold별 점수 계산(저장용) ---------
    max_fold = int(fold_ids.max())
    f1_per_fold, acc_per_fold, auroc_per_fold, auprc_per_fold, chal_per_fold = [], [], [], [], []
    for k in range(1, max_fold + 1):
        mask = (fold_ids == k)
        y_k = y[mask]
        p_k = oof_ens[mask]
        if y_k.size == 0:
            continue
        yhat_k = (p_k >= thr_f1).astype(int)
        f1_per_fold.append(compute_f_measure(y_k, yhat_k))
        acc_per_fold.append(compute_accuracy(y_k, yhat_k))
        auroc_k, auprc_k = compute_auc(y_k, p_k)
        auroc_per_fold.append(auroc_k)
        auprc_per_fold.append(auprc_k)
        chal_per_fold.append(compute_challenge_score(y_k, p_k))

    scores_dict = {
        "f1": np.asarray(f1_per_fold, dtype=float),
        "accuracy": np.asarray(acc_per_fold, dtype=float),
        "auroc": np.asarray(auroc_per_fold, dtype=float),
        "auprc": np.asarray(auprc_per_fold, dtype=float),
        "challenge_score": np.asarray(chal_per_fold, dtype=float),
    }

    # --------- 최종 학습(전체 데이터) ---------
    if verbose:
        print("Fitting final models on all data ...")
    X_res, y_res = smote_fit_resample(Xs, y, seed=RNG)
    fitted: Dict[str, object] = {}

    rf = build_rf()
    rf.fit(X_res, y_res)
    fitted["rf"] = rf

    xgb = build_xgb(y_res)
    xgb.fit(X_res, y_res)
    fitted["xgb"] = xgb

    lr = build_lr()
    lr.fit(X_res, y_res)
    fitted["lr"] = lr

    os.makedirs(model_folder, exist_ok=True)
    payload = {
        "scaler": scaler,
        "kind": "rf+xgb+lr",
        "members": members,
        "weights": weights,
        "threshold_f1": float(thr_f1),
        "threshold_challenge": float(thr_chal),
        "models": fitted,
        "feature_dim": int(X.shape[1]),
        # ----- 재학습 없이 p-value 산출용 -----
        "oof_label": y.astype(int),
        "oof_proba": oof_ens.astype(float),
        "fold_ids": fold_ids.astype(int),
        "scores": scores_dict,
    }
    save_payload(model_folder, payload)
    if verbose:
        print("Training complete.")

# =========================================================
# 로드/추론
# =========================================================
def load_model(model_folder: str, verbose: bool):
    return load_payload(model_folder, verbose=verbose)

def _predict_proba(models: Dict[str, object], members: List[str], weights: Dict[str, float], X: np.ndarray) -> float:
    total = 0.0
    for m in members:
        p = float(models[m].predict_proba(X)[0, 1])
        total += weights[m] * p
    return float(total)

def run_model(record: str, model, verbose: bool):
    f = extract_features(record).reshape(1, -1)
    X = model["scaler"].transform(f)
    proba = _predict_proba(model["models"], model["members"], model["weights"], X)
    t = float(model.get("threshold_challenge", 0.5))
    binary = int(proba >= t)
    return binary, proba
