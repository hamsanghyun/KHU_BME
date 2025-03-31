#패스필터 설계
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered
    
#리드별 각각 적용
raw = ecg_data[1]  # 리드 II
filtered = bandpass_filter(raw, fs=400)

plt.plot(raw, label="Raw", alpha=0.5)
plt.plot(filtered, label="Filtered", linewidth=1.5)
plt.legend()
plt.title("Lead II - Before & After Filtering")
plt.show()


#리드 자동적용
# 리드 선택
lead_signal = ecg_data[lead_index]

# 필터링 적용
lead_signal = bandpass_filter(lead_signal, FS)
