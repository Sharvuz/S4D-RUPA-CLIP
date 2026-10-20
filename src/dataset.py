"""
Dataset loaders for UCF-Crime and XD-Violence with CLIP features.
Includes high-speed Radar Cache resolution for Kaggle (/kaggle/input), Linux, and local environments.
Pre-resolves all paths during Dataset initialization to eliminate runtime I/O search bottlenecks.
Seamlessly handles bidirectional character normalizations (# <-> + <-> _).
"""

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import tools


_NPY_RADAR_CACHE = {}


def init_radar_cache():
    """
    Scans the system storage once at startup to index all .npy feature files.
    Maps file basenames and character variants (# <-> + <-> _) to their absolute paths.
    """
    global _NPY_RADAR_CACHE
    if _NPY_RADAR_CACHE:
        return _NPY_RADAR_CACHE

    search_roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path(".."),
        Path("."),
    ]
    print("\n[Radar] Scanning system storage for .npy feature files, please wait...", flush=True)
    t0 = time.time()
    count = 0
    for root in search_roots:
        if root.exists():
            try:
                for p in root.rglob("*.npy"):
                    p_str = str(p.resolve())
                    name = p.name
                    if name not in _NPY_RADAR_CACHE:
                        _NPY_RADAR_CACHE[name] = p_str
                        count += 1
                    # Register character sanitization variants
                    if '#' in name:
                        _NPY_RADAR_CACHE[name.replace('#', '+')] = p_str
                        _NPY_RADAR_CACHE[name.replace('#', '_')] = p_str
                        _NPY_RADAR_CACHE[name.replace('#', '')] = p_str
                    if '+' in name:
                        _NPY_RADAR_CACHE[name.replace('+', '#')] = p_str
                        _NPY_RADAR_CACHE[name.replace('+', '_')] = p_str
                        _NPY_RADAR_CACHE[name.replace('+', '')] = p_str
            except Exception:
                pass

    t1 = time.time()
    print(f"[Radar] Indexing complete: {count} unique feature files located in {t1 - t0:.2f}s!\n", flush=True)
    return _NPY_RADAR_CACHE


def find_dataset_root(hint: str) -> str:
    """Helper for notebook diagnostics."""
    cache = init_radar_cache()
    hint_lower = hint.lower()
    for name, path in cache.items():
        if hint_lower in path.lower():
            return str(Path(path).parent)
    return ""


def resolve_npy_path(path_str: str) -> str:
    """
    Translates hardcoded paths from list CSVs into valid absolute paths
    via the pre-indexed Radar Cache in O(1) time.
    """
    if os.path.exists(path_str):
        return path_str

    if not _NPY_RADAR_CACHE:
        init_radar_cache()

    basename = os.path.basename(str(path_str).replace('\\', '/'))
    if basename in _NPY_RADAR_CACHE:
        return _NPY_RADAR_CACHE[basename]

    # Check bidirectional hashtag/plus/underscore variations
    variants = [
        basename.replace('#', '+'),
        basename.replace('+', '#'),
        basename.replace('#', '_'),
        basename.replace('#', ''),
        basename.replace('+', ''),
    ]
    for alt in variants:
        if alt in _NPY_RADAR_CACHE:
            return _NPY_RADAR_CACHE[alt]

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
        if not _NPY_RADAR_CACHE:
            init_radar_cache()

        resolved = []
        missing = 0
        for p in self.df['path']:
            rp = resolve_npy_path(p)
            resolved.append(rp)
            if not os.path.exists(rp):
                missing += 1

        self.df['path'] = resolved
        tag = "Normal Train" if normal else ("Abnormal Train" if not test_mode else "Test")
        if missing == 0:
            sample = os.path.basename(self.df.loc[0, 'path'])
            print(f"[Dataset] UCF {tag} ({len(self.df)} videos): 100% verified on disk! (Sample: {sample})", flush=True)
        else:
            print(f"[Dataset WARNING] UCF {tag}: {missing}/{len(self.df)} files could not be located on disk!", flush=True)

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

        if not _NPY_RADAR_CACHE:
            init_radar_cache()

        resolved = []
        missing = 0
        for p in self.df['path']:
            rp = resolve_npy_path(p)
            resolved.append(rp)
            if not os.path.exists(rp):
                missing += 1

        self.df['path'] = resolved
        tag = "Normal Train" if normal else ("Abnormal Train" if not test_mode else "Test")
        if missing == 0:
            sample = os.path.basename(self.df.loc[0, 'path'])
            print(f"[Dataset] XD {tag} ({len(self.df)} videos): 100% verified on disk! (Sample: {sample})", flush=True)
        else:
            print(f"[Dataset WARNING] XD {tag}: {missing}/{len(self.df)} files could not be located on disk!", flush=True)

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
