import numpy as np
import torch
from torch.utils.data import Dataset

from lunar_fno.utils.io_contracts import canonicalize_spectrograms


class SpectrogramDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = canonicalize_spectrograms(x)
        self.y = np.asarray(y).astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        xi = torch.from_numpy(self.x[idx]).float()
        yi = torch.tensor(self.y[idx]).float()
        return xi, yi
