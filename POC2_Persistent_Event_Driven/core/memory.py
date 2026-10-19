import torch

class PersistentState:
    """Simple exponentially-decaying persistent state.

    This is deliberately transparent for POC2. It is not intended to model
    semantic memory by itself; it provides a controllable memory timescale.
    """
    def __init__(self, dim: int, decay: float = 0.995):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.dim = dim
        self.decay = decay
        self.state = None

    def reset(self, batch_size: int, device=None):
        self.state = torch.zeros(batch_size, self.dim, device=device)

    def update(self, x: torch.Tensor):
        if self.state is None or self.state.shape[0] != x.shape[0]:
            self.reset(x.shape[0], x.device)
        self.state = self.decay * self.state + (1.0 - self.decay) * x
        return self.state
