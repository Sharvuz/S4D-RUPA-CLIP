"""
Dataset loaders for UCF-Crime and XD-Violence with CLIP features.
Includes intelligent path resolution for Kaggle, Linux, and local environments.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import tools


_NPY_RADAR_CACHE = {}


def resolve_npy_path(path_str):
    """
    Intelligent Path Resolver:
    Locates the .npy file automatically on Kaggle (/kaggle/input), local directories,
    or relative paths even if the CSV has hardcoded paths from another machine.
    """
    if os.path.exists(path_str):
        return path_str

    basename = os.path.basename(str(path_str).replace('\\', '/'))
    if basename in _NPY_RADAR_CACHE:
        return _NPY_RADAR_CACHE[basename]

    if not _NPY_RADAR_CACHE:
        search_roots = [
            Path("/kaggle/input"),
            Path("/kaggle/working"),
            Path("./UCFClipFeatures"),
            Path("../UCFClipFeatures"),
            Path("./XDTrainClipFeatures"),
            Path("./XDTestClipFeatures"),
            Path("../XDTrainClipFeatures"),
            Path("../XDTestClipFeatures"),
            Path("."),
        ]
        for root in search_roots:
            if root.exists():
                try:
                    for p in root.rglob("*.npy"):
                        if p.name not in _NPY_RADAR_CACHE:
                            _NPY_RADAR_CACHE[p.name] = str(p.resolve())
                except Exception:
                    pass

    if basename in _NPY_RADAR_CACHE:
        return _NPY_RADAR_CACHE[basename]

    # Check common Kaggle filename normalization variations
    variants = [
        basename.replace('#', '+'),
        basename.replace('#', '_'),
        basename.replace('#', '')
    ]
    for alt in variants:
        if alt in _NPY_RADAR_CACHE:
            return _NPY_RADAR_CACHE[alt]

    return path_str


class UCFDataset(data.Dataset):
    """
    DataLoader for UCF-Crime Video Features (CLIP ViT-B/16, 512-d).
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

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        raw_path = self.df.loc[index]['path']
        label = self.df.loc[index]['label']
        path = resolve_npy_path(raw_path)

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
    Label 'A' or 'Normal' corresponds to normal videos.
    Labels 'B1', 'B2', 'B4', 'B5', 'B6', 'G' correspond to violent events.
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

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        raw_path = self.df.loc[index]['path']
        label = self.df.loc[index]['label']
        path = resolve_npy_path(raw_path)

        clip_feature = np.load(path)

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length, path, label
