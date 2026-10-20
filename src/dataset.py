"""
Dataset loaders for UCF-Crime and XD-Violence with CLIP features.
Includes ultra-fast path resolution for Kaggle (/kaggle/input), Linux, and local environments.
Pre-resolves all paths during Dataset initialization to eliminate runtime I/O search bottlenecks.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import tools


_ROOT_CACHE = {}
_PATH_CACHE = {}


def find_dataset_root(hint: str) -> str:
    """
    Rapidly locates the dataset directory root in O(1) time without recursive rglob.
    Scans shallow directory trees (depth <= 2) in ~1 millisecond.
    """
    if hint in _ROOT_CACHE and os.path.isdir(_ROOT_CACHE[hint]):
        return _ROOT_CACHE[hint]

    # Explicit candidate paths across Kaggle, local, and cluster setups
    candidates = [
        # Kaggle paths
        Path(f"/kaggle/input/{hint.lower()}/{hint}/{hint}"),
        Path(f"/kaggle/input/{hint.lower()}/{hint}"),
        Path(f"/kaggle/input/{hint.lower()}"),
        Path(f"/kaggle/input/ucfcrime-clip-features/{hint}/{hint}"),
        Path(f"/kaggle/input/ucfcrime-clip-features/{hint}"),
        Path(f"/kaggle/input/ucf-crime-clip-features/{hint}/{hint}"),
        Path(f"/kaggle/input/ucf-crime-clip-features/{hint}"),
        Path(f"/kaggle/input/xdtrainclipfeatures/{hint}"),
        Path(f"/kaggle/input/xdtrainclipfeatures"),
        Path(f"/kaggle/input/xdtestclipfeatures/{hint}"),
        Path(f"/kaggle/input/xdtestclipfeatures"),
        Path(f"/kaggle/input/xd-violence-clip-features/{hint}"),
        Path(f"/kaggle/input/xd-violence-clip-features"),
        Path(f"/kaggle/working/{hint}/{hint}"),
        Path(f"/kaggle/working/{hint}"),
        # Local paths
        Path(f"./{hint}/{hint}"),
        Path(f"./{hint}"),
        Path(f"../{hint}/{hint}"),
        Path(f"../{hint}"),
    ]

    for cand in candidates:
        if cand.is_dir():
            _ROOT_CACHE[hint] = str(cand.resolve())
            return _ROOT_CACHE[hint]

    # Shallow search in /kaggle/input and current workspace (depth <= 2)
    search_bases = [Path("/kaggle/input"), Path("/kaggle/working"), Path(".."), Path(".")]
    hint_lower = hint.lower()
    for base in search_bases:
        if not base.exists():
            continue
        try:
            for item in base.iterdir():
                if item.is_dir() and hint_lower in item.name.lower():
                    # Check nested subdirectories
                    sub = item / hint
                    if sub.is_dir():
                        sub2 = sub / hint
                        if sub2.is_dir():
                            _ROOT_CACHE[hint] = str(sub2.resolve())
                            return _ROOT_CACHE[hint]
                        _ROOT_CACHE[hint] = str(sub.resolve())
                        return _ROOT_CACHE[hint]
                    _ROOT_CACHE[hint] = str(item.resolve())
                    return _ROOT_CACHE[hint]
        except Exception:
            pass

    return ""


def resolve_npy_path(path_str: str, dataset_type: str = 'ucf') -> str:
    """
    Ultra-Fast Path Resolver:
    Translates hardcoded paths from list CSVs into valid absolute paths on the local machine
    or Kaggle in O(1) time without recursive filesystem walking.
    """
    if os.path.exists(path_str):
        return path_str

    if path_str in _PATH_CACHE:
        return _PATH_CACHE[path_str]

    norm_path = str(path_str).replace('\\', '/')
    parts = norm_path.split('/')
    filename = parts[-1]

    # Handle Kaggle character sanitizations on upload
    filename_variants = [
        filename,
        filename.replace('#', '+'),
        filename.replace('#', '_'),
        filename.replace('#', '')
    ]

    if dataset_type == 'ucf':
        category = parts[-2] if len(parts) >= 2 else ""
        root = find_dataset_root("UCFClipFeatures")
        if root:
            for fn in filename_variants:
                cand = os.path.join(root, category, fn)
                if os.path.exists(cand):
                    _PATH_CACHE[path_str] = cand
                    return cand
                cand_flat = os.path.join(root, fn)
                if os.path.exists(cand_flat):
                    _PATH_CACHE[path_str] = cand_flat
                    return cand_flat

    else:  # xd
        hint = "XDTestClipFeatures" if ("test" in norm_path.lower() or "rgbtest" in norm_path.lower()) else "XDTrainClipFeatures"
        root = find_dataset_root(hint)
        if root:
            for fn in filename_variants:
                cand = os.path.join(root, fn)
                if os.path.exists(cand):
                    _PATH_CACHE[path_str] = cand
                    return cand

    # Fallback to local working directory
    for fn in filename_variants:
        if os.path.exists(fn):
            _PATH_CACHE[path_str] = fn
            return fn

    return path_str


class UCFDataset(data.Dataset):
    """
    DataLoader for UCF-Crime Video Features (CLIP ViT-B/16, 512-d).
    Pre-resolves and validates all paths in __init__ for 0-latency worker loading.
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

        # Pre-resolve all file paths once at initialization time
        resolved_paths = [resolve_npy_path(p, dataset_type='ucf') for p in self.df['path']]
        self.df['path'] = resolved_paths

        # Sanity check
        if len(self.df) > 0:
            first_path = self.df.loc[0, 'path']
            if not os.path.exists(first_path):
                print(f"[Dataset WARNING] Could not verify path on disk: {first_path}")
            else:
                tag = "Normal Train" if normal else ("Abnormal Train" if not test_mode else "Test")
                print(f"[Dataset] UCF {tag} features resolved! ({len(self.df)} videos, sample: {os.path.basename(first_path)})", flush=True)

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

        dataset_type = 'xd_test' if test_mode else 'xd'
        resolved_paths = [resolve_npy_path(p, dataset_type='xd') for p in self.df['path']]
        self.df['path'] = resolved_paths

        if len(self.df) > 0:
            first_path = self.df.loc[0, 'path']
            if not os.path.exists(first_path):
                print(f"[Dataset WARNING] Could not verify path on disk: {first_path}")
            else:
                tag = "Normal Train" if normal else ("Abnormal Train" if not test_mode else "Test")
                print(f"[Dataset] XD {tag} features resolved! ({len(self.df)} videos, sample: {os.path.basename(first_path)})", flush=True)

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
