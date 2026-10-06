import numpy as np

class DeltaEncoder:
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def encode(self, x):
        if x.ndim != 2:
            raise ValueError("Expected [time, features].")
        delta = np.abs(x - np.roll(x, 1, axis=0))
        delta[0] = 0.0
        return (delta >= self.threshold).astype(np.float32)
