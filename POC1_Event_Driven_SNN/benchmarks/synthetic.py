import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticTemporalDataset(Dataset):
    """Controlled temporal benchmark with adjustable event density."""

    def __init__(self, n_samples, sequence_length, input_dim, n_classes,
                 event_density, noise_std=0.05, seed=42):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, noise_std, (n_samples, sequence_length, input_dim)).astype(np.float32)
        y = rng.integers(0, n_classes, n_samples)

        width = max(2, input_dim // 6)
        for i, cls in enumerate(y):
            start = [sequence_length // 5, sequence_length // 2,
                     (3 * sequence_length) // 5][int(cls) % 3]
            end = min(sequence_length, start + sequence_length // 5)
            lo, hi = int(cls) * width, min(input_dim, (int(cls) + 1) * width)
            x[i, start:end, lo:hi] += 1.0

        mask = rng.random(x.shape) < event_density
        x *= mask.astype(np.float32)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
