import os
import numpy as np

# 処理するディレクトリのパス
directory = '/Users/nksu/Desktop/test/raw2'

for filename in os.listdir(directory):
    if filename.endswith('.npy'):  # npyファイルのみを対象とする
        # ファイルを読み込む
        filepath = os.path.join(directory, filename)
        data = np.load(filepath)

        transposed_data = np.transpose(data)
        
        # 全体の平均と標準偏差を計算する
        overall_mean = np.mean(transposed_data)
        overall_std = np.std(transposed_data)

        # Z-score normalizationを適用する
        normalized_data = (transposed_data - overall_mean) / overall_std

        retransposed_data = np.transpose(normalized_data)
        np.save(filepath, retransposed_data)
        print(f'Processed file saved as: {filename}')

