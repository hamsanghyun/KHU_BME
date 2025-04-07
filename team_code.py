#!/usr/bin/env python

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys

from helper_code import *

# 환경 변수값 불러오기기
N_ESTIMATORS    = int(os.getenv("N_ESTIMATORS", "12"))
MAX_LEAF_NODES  = int(os.getenv("MAX_LEAF_NODES", "34"))
RANDOM_STATE    = int(os.getenv("RANDOM_STATE", "56"))
FS_DEFAULT      = float(os.getenv("FS_DEFAULT", "400"))
BANDPASS_LOWCUT = float(os.getenv("BANDPASS_LOWCUT", "0.5"))
BANDPASS_HIGHCUT= float(os.getenv("BANDPASS_HIGHCUT", "40.0"))
BANDPASS_ORDER  = int(os.getenv("BANDPASS_ORDER", "3"))
THRESHOLD_RATIO = float(os.getenv("THRESHOLD_RATIO", "0.1"))
V1_INDEX        = int(os.getenv("V1_INDEX", "6"))
V2_INDEX        = int(os.getenv("V2_INDEX", "7"))


def train_model(data_folder, model_folder, verbose):
 
    if verbose:
        print('데이터 파일 찾는 중...')
    records = find_records(data_folder)
    num_records = len(records)
    if num_records == 0:
        raise FileNotFoundError('데이터가 제공되지 않았습니다.')

    if verbose:
        print('피처와 라벨 추출 중...')
    # 전체 피처 차원: age (1) + 성별 one-hot (3) + 통계/주파수 기반 피처 (11) + QRS duration (1) = 17
    features = np.zeros((num_records, 17), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')
        record_path = os.path.join(data_folder, records[i])
        features[i] = extract_features(record_path)
        labels[i] = load_label(record_path)
    
    # 학습 데이터와 평가 데이터를 분할 (라벨 분포를 유지)
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)
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


# 피처 추출

from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal


def compute_qrs_duration(ecg, fs, threshold_ratio):
    """
    R-peak를 기준으로 신호의 절대값이 최대값의 일정 비율(threshold_ratio) 이하가 되는 지점.
    """
    if len(ecg) == 0:
        return 0.0
    idx = np.argmax(np.abs(ecg))
    R_val = np.abs(ecg[idx])
    threshold = R_val * threshold_ratio
    left = idx
    while left > 0 and np.abs(ecg[left]) > threshold:
        left -= 1
    right = idx
    while right < len(ecg) - 1 and np.abs(ecg[right]) > threshold:
        right += 1
    duration_samples = right - left
    duration_seconds = duration_samples / fs
    return duration_seconds


def extract_features(record):
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
    # 밴드패스 필터 적용 (lowcut, highcut, order 사용)
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
    
    # QRS duration 계산: 환경 변수로 지정한 V1_INDEX, V2_INDEX를 사용
    if signal.shape[1] > max(V1_INDEX, V2_INDEX):
        qrs_v1 = compute_qrs_duration(signal[:, V1_INDEX], fs, THRESHOLD_RATIO)
        qrs_v2 = compute_qrs_duration(signal[:, V2_INDEX], fs, THRESHOLD_RATIO)
        qrs_duration = (qrs_v1 + qrs_v2) / 2
    else:
        qrs_duration = 0.0

    # 피처 벡터 구성:
    # 순서: age, one_hot_encoding_sex (3), signal_mean, signal_std, signal_max, signal_min,
    # signal_range, signal_median, IQR, energy, zero_crossings, signal_skew, signal_kurt,
    # dominant_frequency, qrs_duration
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
