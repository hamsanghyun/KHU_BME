#!/usr/bin/env python

"""
team_code.py

본 파일은 ECG 데이터를 기반으로 한 Chagas 질환 예측 모델 학습 및 추론을 위한 팀 코드입니다.
아래 기능을 포함합니다:
  - 데이터 전처리 및 피처 추출 (기본 통계 및 추가 피처: QRS duration 등)
  - 데이터 분할 (학습/평가 데이터를 stratify하여 분리)
  - RandomForestClassifier 기반 모델 학습 (하이퍼파라미터는 환경 변수로 설정)
  - 학습된 모델 저장 및 추론 기능

피처 추출 부분은 joblib.Parallel을 활용하여 병렬 처리하였으며,
QRS duration 계산은 Pan-Tompkins 알고리즘을 사용하고, numba를 통해 최적화하였습니다.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import sys
from numba import jit

from helper_code import *

# 환경 변수값 불러오기 (설정 파일이나 환경 변수로 쉽게 조정 가능)
N_ESTIMATORS    = int(os.getenv("N_ESTIMATORS", "12"))
MAX_LEAF_NODES  = int(os.getenv("MAX_LEAF_NODES", "34"))
RANDOM_STATE    = int(os.getenv("RANDOM_STATE", "56"))
FS_DEFAULT      = float(os.getenv("FS_DEFAULT", "400"))
BANDPASS_LOWCUT = float(os.getenv("BANDPASS_LOWCUT", "0.5"))
BANDPASS_HIGHCUT= float(os.getenv("BANDPASS_HIGHCUT", "40.0"))
BANDPASS_ORDER  = int(os.getenv("BANDPASS_ORDER", "3"))
# Pan-Tompkins 임계값 계수; 일반적으로 0.5 정도가 사용됨
THRESHOLD_FACTOR = float(os.getenv("THRESHOLD_FACTOR", "0.5"))
V1_INDEX        = int(os.getenv("V1_INDEX", "6"))
V2_INDEX        = int(os.getenv("V2_INDEX", "7"))

# =============================================================================
# Required functions: train_model, load_model, run_model
# =============================================================================

def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('데이터 파일 찾는 중...')
    records = find_records(data_folder)
    num_records = len(records)
    if num_records == 0:
        raise FileNotFoundError('데이터가 제공되지 않았습니다.')
    
    if verbose:
        print('피처와 라벨 추출 중...')

    features_list = Parallel(n_jobs=-1)(
        delayed(extract_features)(os.path.join(data_folder, rec)) for rec in records
    )
    labels_list = Parallel(n_jobs=-1)(
        delayed(load_label)(os.path.join(data_folder, rec)) for rec in records
    )
    features = np.vstack(features_list)
    labels = np.array(labels_list, dtype=bool)

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    if verbose:
        print(f'학습 데이터 샘플 수: {X_train.shape[0]}, 평가 데이터 샘플 수: {X_val.shape[0]}')
    
    if verbose:
        print('모델 학습 중...')
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_leaf_nodes=MAX_LEAF_NODES,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ).fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    accuracy = (y_pred == y_val).mean()
    if verbose:
        print(f'평가 데이터 정확도: {accuracy:.3f}')
    
    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model)
    
    if verbose:
        print('학습 완료.\n')


def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model


def run_model(record, model, verbose):
    model = model['model']
    features = extract_features(record)
    features = features.reshape(1, -1)
    probability = model.predict_proba(features)[0][1]
    binary_output = int(probability > 0.3)
    return binary_output, probability


# =============================================================================
# Optional functions: 피처 추출 관련 기능
# =============================================================================

from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal

@jit(nopython=True)
def compute_qrs_duration_pantompkins(ecg, fs, threshold_factor):
    """
    Pan-Tompkins 알고리즘을 활용하여 단일 1차원 ECG 신호의 QRS duration (초)을 계산합니다.
    단계:
      1. 미분 (연속 차분)
      2. 제곱
      3. 이동 창 적분 (창 크기: 150ms)
      4. 통합 신호에서 R-peak 검출 및 threshold_factor를 이용하여 QRS onset/offset 찾기
    최적화를 위해 numba의 JIT 컴파일을 적용합니다.
    """
    n = len(ecg)
    # 1. 미분
    diff_ecg = np.empty(n)
    diff_ecg[0] = 0.0
    for i in range(1, n):
        diff_ecg[i] = ecg[i] - ecg[i-1]
    # 2. 제곱
    squared = diff_ecg * diff_ecg
    # 3. 이동 창 적분: 창 크기 = 150ms
    window_size = int(0.150 * fs)
    integrated = np.empty(n)
    half_window = window_size // 2
    for i in range(n):
        start = i - half_window if i - half_window > 0 else 0
        end = i + half_window if i + half_window < n else n
        s = 0.0
        for j in range(start, end):
            s += squared[j]
        integrated[i] = s / (end - start)
    # 4. R-peak 검출: 통합 신호의 최대값 위치
    r_index = 0
    max_val = integrated[0]
    for i in range(1, n):
        if integrated[i] > max_val:
            max_val = integrated[i]
            r_index = i
    # 임계값 설정 (threshold_factor * max값)
    threshold = threshold_factor * max_val
    # QRS onset: r_index에서 왼쪽으로, threshold 아래가 될 때까지 이동
    onset = r_index
    while onset > 0 and integrated[onset] > threshold:
        onset -= 1
    # QRS offset: r_index에서 오른쪽으로, threshold 아래가 될 때까지 이동
    offset = r_index
    while offset < n - 1 and integrated[offset] > threshold:
        offset += 1
    duration_samples = offset - onset
    duration_seconds = duration_samples / fs
    return duration_seconds

def extract_features(record):
    """
    주어진 ECG 기록에서 다음 피처들을 추출하여 17차원 feature vector를 구성합니다.
      - 나이, 성별(one-hot 인코딩)
      - 신호의 평균, 표준편차, 최대, 최소, 범위, 중앙값, IQR, 에너지, 제로 크로싱 수,
        왜도, 첨도, FFT를 통한 dominant frequency, QRS duration (리드 V1, V2의 평균)
    """
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)
    one_hot_encoding_sex = np.zeros(3, dtype=np.float32)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1.0
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1.0
    else:
        one_hot_encoding_sex[2] = 1.0
    
    signal, fields = load_signals(record)
    fs = fields.get("fs", FS_DEFAULT)
    filtered_signal = bandpass_filter(signal, BANDPASS_LOWCUT, BANDPASS_HIGHCUT, fs, BANDPASS_ORDER)
    
    flat_signal = signal.flatten()
    if np.sum(np.isfinite(flat_signal)) > 0:
        signal_mean = np.nanmean(flat_signal)
        signal_std = np.nanstd(flat_signal)
        signal_max = np.nanmax(flat_signal)
        signal_min = np.nanmin(flat_signal)
    else:
        signal_mean = signal_std = signal_max = signal_min = 0.0

    signal_range = signal_max - signal_min
    signal_median = np.nanmedian(flat_signal)
    Q1 = np.percentile(flat_signal, 25)
    Q3 = np.percentile(flat_signal, 75)
    IQR = Q3 - Q1
    energy = np.sum(flat_signal ** 2) / len(flat_signal) if len(flat_signal) > 0 else 0.0
    zero_crossings = np.sum(np.diff(np.sign(flat_signal)) != 0)
    
    from scipy.stats import skew, kurtosis
    signal_skew = skew(flat_signal)
    signal_kurt = kurtosis(flat_signal)
    
    fft_values = np.fft.rfft(flat_signal)
    fft_magnitudes = np.abs(fft_values)
    dominant_frequency = np.argmax(fft_magnitudes)
    
    # QRS duration 계산: 환경 변수로 지정한 V1_INDEX, V2_INDEX를 사용하여 Pan-Tompkins 알고리즘 적용
    if signal.shape[1] > max(V1_INDEX, V2_INDEX):
        qrs_v1 = compute_qrs_duration_pantompkins(signal[:, V1_INDEX], fs, THRESHOLD_FACTOR)
        qrs_v2 = compute_qrs_duration_pantompkins(signal[:, V2_INDEX], fs, THRESHOLD_FACTOR)
        qrs_duration = (qrs_v1 + qrs_v2) / 2
    else:
        qrs_duration = 0.0

    features = np.concatenate((
        [age],
        one_hot_encoding_sex,
        [signal_mean, signal_std, signal_max, signal_min,
         signal_range, signal_median, IQR, energy, zero_crossings,
         signal_skew, signal_kurt, dominant_frequency, qrs_duration]
    ))
    
    return np.asarray(features, dtype=np.float32)


def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
