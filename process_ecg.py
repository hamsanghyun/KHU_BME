#!/usr/bin/env python
import numpy as np
import os
import scipy as sp
import sys
import wfdb

### Challenge variables
age_string = '# Age:'
sex_string = '# Sex:'
label_string = '# Chagas label:'
probability_string = '# Chagas probability:'

### Challenge data I/O functions

def find_records(folder, header_extension='.hea'):
    records = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == header_extension:
                record = os.path.relpath(os.path.join(root, file), folder)[:-len(header_extension)]
                records.add(record)
    records = sorted(records)
    return records

def load_header(record):
    header_file = get_header_file(record)
    header = load_text(header_file)
    return header

def load_signals(record):
    signal, fields = wfdb.rdsamp(record)
    return signal, fields

def load_label(record):
    header = load_header(record)
    label = get_label(header)
    return label

def load_probability(record):
    header = load_header(record)
    probability = get_probability(header, allow_missing=True)
    return probability

def save_outputs(output_file, record_name, label, probability):
    output_string = f'{record_name}\n{label_string} {label}\n{probability_string} {probability}\n'
    save_text(output_file, output_string)
    return output_string

### 추가로 필요한 함수들

def get_header_file(record):
    if not record.endswith('.hea'):
        return record + '.hea'
    else:
        return record

def load_text(filename):
    with open(filename, 'r') as f:
        string = f.read()
    return string

def get_label(header_string, allow_missing=False):
    label = None
    for line in header_string.split('\n'):
        if line.startswith(label_string):
            label = line[len(label_string):].strip()
            break
    if label is None and not allow_missing:
        raise Exception('헤더에 라벨 정보가 없습니다.')
    return label

def get_probability(header_string, allow_missing=True):
    probability = None
    for line in header_string.split('\n'):
        if line.startswith(probability_string):
            probability = line[len(probability_string):].strip()
            break
    # allow_missing=True이면 확률 정보가 없을 경우 그냥 None 반환
    return probability

def save_text(filename, string):
    with open(filename, 'w') as f:
        f.write(string)

### 입력 및 출력 폴더 경로 지정
INPUT_FOLDER = "C:\\Users\hamsa\OneDrive\바탕 화면\Physionet\output_folder"
OUTPUT_FOLDER = "C:\\Users\hamsa\OneDrive\바탕 화면\Physionet\data_folder"

def main():
    # 출력 폴더가 없으면 생성
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # 입력 폴더에서 모든 기록(.hea 파일 기준) 찾기
    records = find_records(INPUT_FOLDER)
    print(f"총 {len(records)}개의 기록을 찾았습니다.")
    
    # 각 기록에 대해 데이터 로드 및 저장
    for rec in records:
        record_path = os.path.join(INPUT_FOLDER, rec)
        print(f"\nProcessing record: {record_path}")
        try:
            header = load_header(record_path)
            signal, fields = load_signals(record_path)
            label = load_label(record_path)
            probability = load_probability(record_path)
            
            print("헤더 내용:")
            print(header)
            print("신호 데이터 shape:", signal.shape)
            print("필드 정보:", fields)
            print("라벨:", label)
            print("확률:", probability)
            
            # 출력 폴더에 결과 저장 (텍스트 파일)
            record_name = os.path.basename(rec)
            output_file = os.path.join(OUTPUT_FOLDER, f"{record_name}_output.txt")
            save_outputs(output_file, record_name, label, probability)
            print(f"출력 정보 저장 완료: {output_file}")

            # ECG 신호 데이터를 CSV 파일로 저장
            ecg_file = os.path.join(OUTPUT_FOLDER, f"{record_name}_ecg.csv")
            print(f"ECG 데이터를 저장할 파일 경로: {ecg_file}")
            print(f"signal 타입: {type(signal)}, shape: {signal.shape}")
            np.savetxt(ecg_file, signal, delimiter=",")
            print(f"ECG 신호 데이터 저장 완료: {ecg_file}")
        except Exception as e:
            print(f"Record {rec} 처리 중 에러 발생: {e}")

if __name__ == '__main__':
    main()