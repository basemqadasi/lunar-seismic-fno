from pathlib import Path
from typing import Tuple

import numpy as np


def load_npz(path: str) -> np.lib.npyio.NpzFile:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    return np.load(p, allow_pickle=True)


def canonicalize_waveforms(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    while arr.ndim > 2 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected waveform_data to become (N, L), got {arr.shape}")
    return arr.astype(np.float32)


def canonicalize_spectrograms(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.ndim != 4 or arr.shape[-1] != 1:
        raise ValueError(f"Expected spectrogram_data to become (N, H, W, 1), got {arr.shape}")
    return arr.astype(np.float32)


def get_required_arrays(npz: np.lib.npyio.NpzFile, x_key: str, y_key: str) -> Tuple[np.ndarray, np.ndarray]:
    if x_key not in npz.files:
        raise KeyError(f"Missing key '{x_key}'. Available keys: {npz.files}")
    if y_key not in npz.files:
        raise KeyError(f"Missing key '{y_key}'. Available keys: {npz.files}")
    return npz[x_key], npz[y_key]
