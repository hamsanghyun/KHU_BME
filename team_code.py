#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    features = np.zeros((num_records, 16), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features[i] = extract_features(record)
        labels[i] = load_label(record)

    # Train the models.
    if verbose:
        print('Training the model on the data...')

    # This very simple model trains a random forest model with very simple features.

    # Define the parameters for the random forest classifier and regressor.
    n_estimators = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state = 56  # Random state; set for reproducibility.

    # Fit the model.
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state, class_weight="balanced").fit(features, labels)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    model = model['model']

    # Extract the features.
    features = extract_features(record)
    features = features.reshape(1, -1)

    # 예측 확률 계산 (양성 클래스에 대한 확률)
    probability = model.predict_proba(features)[0][1]
    
    # 지정된 임계값 (예: 0.3) 이상이면 양성으로 예측
    binary_output = int(probability > 0.3)
    
    return binary_output, probability
################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# bandpass_filter
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)
    
    # One-hot encoding for sex: [Female, Male, Unknown]
    one_hot_encoding_sex = np.zeros(3, dtype=np.float32)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1.0
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1.0
    else:
        one_hot_encoding_sex[2] = 1.0

    # Load the ECG signal (2D array: samples x leads)
    signal, fields = load_signals(record)
    # 샘플링 주파수 얻기 (fields에 포함되어 있다고 가정)
    fs = fields.get("fs", 250)  # 기본값 250Hz

    # 여기서 밴드패스 필터 적용 (예: 0.5Hz ~ 40Hz)
    filtered_signal = bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=fs, order=3)
    
    # Flatten the signal across all leads for global statistics
    flat_signal = signal.flatten()
    if np.sum(np.isfinite(flat_signal)) > 0:
        signal_mean = np.nanmean(flat_signal)
        signal_std = np.nanstd(flat_signal)
        signal_max = np.nanmax(flat_signal)
        signal_min = np.nanmin(flat_signal)
    else:
        signal_mean = signal_std = signal_max = signal_min = 0.0

    # 추가 피처: 범위, 중앙값, 사분위 범위, 에너지, 제로 크로싱, 왜도, 첨도
    signal_range = signal_max - signal_min
    signal_median = np.nanmedian(flat_signal)
    Q1 = np.percentile(flat_signal, 25)
    Q3 = np.percentile(flat_signal, 75)
    IQR = Q3 - Q1
    energy = np.sum(flat_signal ** 2) / len(flat_signal) if len(flat_signal) > 0 else 0.0
    zero_crossings = np.sum(np.diff(np.sign(flat_signal)) != 0)
    
    # 왜도와 첨도 계산 (scipy.stats 사용)
    from scipy.stats import skew, kurtosis
    signal_skew = skew(flat_signal)
    signal_kurt = kurtosis(flat_signal)
    
     # 주파수 도메인 피처: FFT를 통해 가장 큰 진폭을 가진 주파수의 인덱스 (샘플링 주파수로 정규화할 수 있음)
    fft_values = np.fft.rfft(flat_signal)
    fft_magnitudes = np.abs(fft_values)
    dominant_frequency = np.argmax(fft_magnitudes)  # dominant frequency index
    
    # Feature 벡터 구성:
    # 순서: age, one-hot sex(3), signal_mean, signal_std, signal_max, signal_min,
    # signal_range, signal_median, IQR, energy, zero_crossings, signal_skew, signal_kurt
    features = np.concatenate((
        [age],
        one_hot_encoding_sex,
        [signal_mean, signal_std, signal_max, signal_min, signal_range, signal_median, IQR, energy, zero_crossings, signal_skew, signal_kurt, dominant_frequency]
    ))
    
    return np.asarray(features, dtype=np.float32)


# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)