import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = torch.from_numpy(np.load(X_path))
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            
            return X, y, subject_idx
        else:
            return X, subject_idx
        
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]
'''
train_dataset = ThingsMEGDataset(split="test", data_dir="/Volumes/Extreme Pro/data")
val_dataset = ThingsMEGDataset(split="val", data_dir="/Volumes/Extreme Pro/data")
test_dataset = ThingsMEGDataset(split="test", data_dir="/Volumes/Extreme Pro/data")

print(f"Number of samples in train set: {len(train_dataset)}")
print(f"Number of samples in val set: {len(val_dataset)}")
print(f"Number of samples in test set: {len(test_dataset)}")

train_sample= train_dataset[0]
val_sample= val_dataset[0]
test_sample= test_dataset[0]

print(f"First train sample: {train_sample}")
print(f"First val sample: {val_sample}")
print(f"First test sample: {test_sample}")
'''