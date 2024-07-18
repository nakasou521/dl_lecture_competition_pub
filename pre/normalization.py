import os
import numpy as np

# 処理対象のディレクトリ
directory = '/Users/nksu/Desktop/深層学習/final/data-omni/val_y'

# 新しいファイルの保存先ディレクトリ
output_directory = '/Users/nksu/Desktop/深層学習/final/normalized-data-omni/val_y'

# ディレクトリが存在しない場合は作成する
os.makedirs(output_directory, exist_ok=True)

# ディレクトリ内のファイルを走査
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

        # 新しいファイル名を生成して保存する
        output_filename = os.path.join(output_directory, f'{filename}')
        np.save(output_filename, retransposed_data)
        
        print(f'Processed file saved as: {output_filename}')
