import numpy as np
import matplotlib.pyplot as plt

# サンプリングレートとデータ長さを定義する（例として）
sampling_rate = 200  # サンプリングレート 200 Hz
data_length = 281    # データの長さ

# npyファイルからデータを読み込む
file_path = '/Users/nksu/Library/CloudStorage/Dropbox/深層学習/final/data-omni/train_X/00000.npy'
waveform_data = np.load(file_path)

transposed_data = np.transpose(waveform_data)

# 時間軸を生成する
time = np.arange(data_length) / sampling_rate

plt.figure(figsize=(10, 4))
plt.plot(time, transposed_data)
plt.title('Waveform Data')
plt.xlabel('tmie [s]')
plt.ylabel('Amplitude')
plt.xlim(0, 1.4)
plt.grid(False)
plt.show()