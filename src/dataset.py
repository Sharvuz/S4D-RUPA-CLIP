"""
Dataset loaders for UCF-Crime and XD-Violence with CLIP features.
High-speed data feeding for standalone S4D-RUPA-CLIP.
Seamlessly loads pre-mapped CSVs from Kaggle artifacts or falls back to auto-resolution.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import tools


_CACHE = {}


def resolve_npy_path(path_str: str) -> str:
    """Fallback resolver if the CSV still contains raw/foreign paths."""
    if os.path.exists(path_str):
        return path_str

    if path_str in _CACHE:
        return _CACHE[path_str]

    basename = os.path.basename(str(path_str).replace('\\', '/'))

    # Quick scan in common Kaggle and local roots if not cached
    if not _CACHE:
        for root_dir in ["/kaggle/input", "/kaggle/working", "..", "."]:
            if os.path.exists(root_dir):
                for r, _, files in os.walk(root_dir, followlinks=False):
                    for f in files:
                        if f.endswith('.npy'):
                            fp = os.path.join(r, f)
                            _CACHE[f] = fp
                            if '#' in f:
                                _CACHE[f.replace('#', '+')] = fp
                                _CACHE[f.replace('#', '_')] = fp
                                _CACHE[f.replace('#', '')] = fp
                            if '+' in f:
                                _CACHE[f.replace('+', '#')] = fp
                                _CACHE[f.replace('+', '_')] = fp
                                _CACHE[f.replace('+', '')] = fp

    if basename in _CACHE:
        return _CACHE[basename]

    for alt in [
        basename.replace('#', '+'),
        basename.replace('+', '#'),
        basename.replace('#', '_'),
        basename.replace('#', ''),
        basename.replace('+', '')
    ]:
        if alt in _CACHE:
            return _CACHE[alt]

    return path_str


class UCFDataset(data.Dataset):
    """
    DataLoader for UCF-Crime Video Features (CLIP ViT-B/16, 512-d).
    Directly loads verified feature paths for 0-latency training.
    """
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool = False, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.normal = normal

        if not test_mode:
            if normal:
                self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
            else:
                self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

        # Fast resolution check: only resolve if the first path doesn't exist
        if len(self.df) > 0 and not os.path.exists(str(self.df.loc[0, 'path'])):
            self.df['path'] = [resolve_npy_path(p) for p in self.df['path']]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        path = self.df.loc[index]['path']
        label = self.df.loc[index]['label']

        clip_feature = np.load(path)

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length, path, label


class XDDataset(data.Dataset):
    """
    DataLoader for XD-Violence Video Features (CLIP ViT-B/16, 512-d).
    Directly loads verified feature paths for 0-latency training.
    """
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool = False, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.normal = normal

        if not test_mode:
            is_normal = self.df['label'].astype(str).str.strip().isin(['A', 'Normal', 'normal'])
            if normal:
                self.df = self.df.loc[is_normal].reset_index(drop=True)
            else:
                self.df = self.df.loc[~is_normal].reset_index(drop=True)

        if len(self.df) > 0 and not os.path.exists(str(self.df.loc[0, 'path'])):
            self.df['path'] = [resolve_npy_path(p) for p in self.df['path']]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        path = self.df.loc[index]['path']
        label = self.df.loc[index]['label']

        clip_feature = np.load(path)

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length, path, label
