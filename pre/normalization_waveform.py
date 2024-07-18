import numpy as np
import matplotlib.pyplot as plt

# サンプリングレートとデータ長さを定義する（例として）
sampling_rate = 200  # サンプリングレート 200 Hz
data_length = 281    # データの長さ

# npyファイルからデータを読み込む
file_path = '/Users/nksu/Library/CloudStorage/Dropbox/深層学習/final/data-omni/train_X/00000.npy'
data = np.load(file_path)

transposed_data = np.transpose(data)

# 全体の平均と標準偏差を計算する
overall_mean = np.mean(transposed_data)
overall_std = np.std(transposed_data)

# Z-score normalizationを適用する
normalized_data = (transposed_data - overall_mean) / overall_std

# 時間軸を生成する
time = np.arange(data_length) / sampling_rate

plt.figure(figsize=(10, 4))
plt.plot(time, normalized_data)
plt.title('Waveform Data')
plt.xlabel('tmie [s]')
plt.ylabel('Amplitude')
plt.xlim(0, 1.4)
plt.grid(False)
plt.show()
