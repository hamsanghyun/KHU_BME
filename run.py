#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 데이터가 저장된 폴더 경로 (CSV 파일과 라벨이 들어있는 텍스트 파일이 함께 있음)
DATA_FOLDER = "C:\\Users\hamsa\OneDrive\바탕 화면\Physionet\data_folder"

# 텍스트 파일에서 라벨을 읽는 함수
def load_label_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    # 예시: 첫 줄은 기록 이름, 두 번째 줄에 "# Chagas label: False" 형태의 라벨이 있음
    label_line = [line for line in lines if line.startswith('# Chagas label:')]
    if label_line:
        label_str = label_line[0].split(':')[1].strip()
        # 예: "False" -> 0, "True" -> 1
        label = 1 if label_str.lower() == 'true' else 0
        return label
    else:
        return None

# ECG CSV 파일 중 하나를 불러와 시각화하는 함수
def visualize_ecg(sample_csv_path):
    ecg_signal = np.loadtxt(sample_csv_path, delimiter=",")
    plt.figure(figsize=(12, 4))
    plt.plot(ecg_signal)
    plt.title(f'ECG Signal from {os.path.basename(sample_csv_path)}')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()

# ECG 신호로부터 특징을 추출하는 함수
def extract_features(ecg_signal, fs=360):
    # ecg_signal이 2차원 배열이라면, 첫번째 채널 사용
    if ecg_signal.ndim > 1:
        ecg_signal = ecg_signal[:, 0]
    features = {}
    features['mean'] = np.mean(ecg_signal)
    features['std'] = np.std(ecg_signal)
    features['max'] = np.max(ecg_signal)
    features['min'] = np.min(ecg_signal)
    # 심박수 추정을 위한 R-피크 검출
    # scipy의 find_peaks 사용 (distance 인자는 샘플링 주파수와 대략적인 최소 심박 간격을 고려)
    peaks, _ = find_peaks(ecg_signal, distance=fs * 0.6)  # 약 100bpm 이하로 가정
    features['num_peaks'] = len(peaks)
    duration_seconds = len(ecg_signal) / fs
    if duration_seconds > 0:
        features['heart_rate'] = len(peaks) / duration_seconds * 60  # bpm
    else:
        features['heart_rate'] = 0
    return features

def main():
    # DATA_FOLDER 내의 모든 CSV 파일(ECG 신호)와 텍스트 파일(라벨 정보) 목록을 확인
    files = os.listdir(DATA_FOLDER)
    csv_files = [f for f in files if f.endswith('_ecg.csv')]
    txt_files = [f for f in files if f.endswith('_output.txt')]
    
    if not csv_files:
        print("CSV 파일이 없습니다.")
        return
    if not txt_files:
        print("라벨 정보가 있는 텍스트 파일이 없습니다.")
        return

    # 1. 샘플 ECG 신호 시각화 (첫 번째 CSV 파일 사용)
    sample_csv = os.path.join(DATA_FOLDER, csv_files[0])
    print(f"샘플 ECG 파일: {sample_csv}")
    visualize_ecg(sample_csv)
    
    # 2. 모든 기록에 대해 특징 추출 및 라벨 읽기
    features_list = []
    labels_list = []
    
    # 텍스트 파일은 recordname_output.txt 형태, CSV 파일은 recordname_ecg.csv 형태로 가정
    for txt_file in txt_files:
        txt_path = os.path.join(DATA_FOLDER, txt_file)
        label = load_label_from_txt(txt_path)
        # record name: txt 파일명의 앞부분 (예: "147242" from "147242_output.txt")
        record_name = txt_file.split('_')[0]
        csv_file = f"{record_name}_ecg.csv"
        csv_path = os.path.join(DATA_FOLDER, csv_file)
        if os.path.exists(csv_path):
            ecg_signal = np.loadtxt(csv_path, delimiter=",")
            feats = extract_features(ecg_signal)
            features_list.append(feats)
            labels_list.append(label)
        else:
            print(f"CSV 파일 {csv_file} 없음.")
    
    # DataFrame으로 변환
    features_df = pd.DataFrame(features_list)
    labels_series = pd.Series(labels_list, name='label')
    
    print("추출된 특징 (상위 5개 기록):")
    print(features_df.head())
    print("라벨 (상위 5개):")
    print(labels_series.head())
    
    # 3. 분류 모델 학습 (예시: Logistic Regression)
    X = features_df.values
    y = labels_series.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("분류 정확도:", acc)
    print("분류 리포트:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
