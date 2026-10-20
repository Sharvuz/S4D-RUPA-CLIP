import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import tools


class UCFDataset(Dataset):
    """
    UCF-Crime Dataset Loader for DMAE-VAD (Pre-extracted CLIP Visual Features).
    """
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool = False, normal: bool = False):
        super().__init__()
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.normal = normal

        if not self.test_mode:
            if self.normal:
                self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
            else:
                self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row['path']
        label = row['label']

        clip_feature = np.load(path)
        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            return torch.from_numpy(clip_feature).float(), label, clip_length
        else:
            split_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            return torch.from_numpy(split_feature).float(), label, clip_length, path, label


class XDDataset(Dataset):
    """
    XD-Violence Dataset Loader for DMAE-VAD.
    """
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool = False, normal: bool = False):
        super().__init__()
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.normal = normal

        if not self.test_mode:
            if self.normal:
                self.df = self.df.loc[self.df['label'].isin(['A', 'Normal'])].reset_index(drop=True)
            else:
                self.df = self.df.loc[~self.df['label'].isin(['A', 'Normal'])].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row['path']
        label = row['label']

        clip_feature = np.load(path)
        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            return torch.from_numpy(clip_feature).float(), label, clip_length
        else:
            split_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            return torch.from_numpy(split_feature).float(), label, clip_length, path, label
