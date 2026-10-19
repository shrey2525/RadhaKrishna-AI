import torch
from torch.utils.data import Dataset

class DelayedRecallDataset(Dataset):
    """Relevant class appears early; after a long gap, the final query asks for it.

    Distractors are random events that should not overwrite the useful memory.
    """
    def __init__(self, n_samples, sequence_length=100, input_dim=16, n_classes=4,
                 memory_gap=50, distractor_probability=0.15, seed=42):
        self.n_samples = n_samples
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.memory_gap = memory_gap
        self.distractor_probability = distractor_probability
        g = torch.Generator().manual_seed(seed)
        self.x = torch.zeros(n_samples, sequence_length, input_dim)
        self.y = torch.randint(0, n_classes, (n_samples,), generator=g)
        self._build(g)

    def _build(self, g):
        for i in range(self.n_samples):
            cls = int(self.y[i])
            # Encode class as a sparse one-hot event near the beginning.
            self.x[i, 0, cls] = 1.0
            # Distractors occur after the memory event.
            for t in range(1, self.sequence_length - 1):
                if torch.rand((), generator=g).item() < self.distractor_probability:
                    idx = int(torch.randint(0, self.input_dim, (1,), generator=g))
                    self.x[i, t, idx] = 1.0
            # Query occurs after the requested memory gap.
            query_t = min(self.sequence_length - 1, self.memory_gap + 1)
            self.x[i, query_t, self.input_dim - 1] = 1.0

    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.x[idx], self.y[idx]
