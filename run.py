import h5py
import os
import numpy as np

# 입력: HDF5 파일 경로
INPUT_FOLDER = r"C:\Users\luhan\OneDrive\바탕 화면\ECG_hdf5"
OUTPUT_FOLDER = r"C:\Users\luhan\OneDrive\바탕 화면\ECG_hdf5"

# 파일 반복
for fname in os.listdir(INPUT_FOLDER):
    if fname.endswith(".hdf5") or fname.endswith(".h5"):
        file_path = os.path.join(INPUT_FOLDER, fname)
        print(f"📂 Processing {fname} ...")

        with h5py.File(file_path, 'r') as f:
            exam_ids = list(f['exam_id'])  # ECG ID
            tracings = f['tracings']       # ECG signal: (N, 5000, 12)
            num = len(exam_ids)

            for i in range(num):
                exam_id = str(exam_ids[i])
                signal = np.array(tracings[i])  # (5000, 12)
                
                # CSV 저장
                csv_filename = os.path.join(OUTPUT_FOLDER, f"{exam_id}_ecg.csv")
                np.savetxt(csv_filename, signal, delimiter=",")
                
                # 예시 라벨 저장 (지금은 임시로 0 저장함 — 나중에 Chagas 여부에 따라 수정 가능)
                txt_filename = os.path.join(OUTPUT_FOLDER, f"{exam_id}_output.txt")
                with open(txt_filename, 'w') as ftxt:
                    ftxt.write(f"{exam_id}\n")
                    ftxt.write("# Chagas label: False\n")
                    ftxt.write("# Chagas probability: 0.0\n")
                
                print(f"✅ Saved: {exam_id}_ecg.csv / {exam_id}_output.txt")
