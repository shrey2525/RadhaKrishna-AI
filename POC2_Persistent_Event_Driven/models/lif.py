import torch
from torch import nn

class LIFCell(nn.Module):
    """Minimal differentiable LIF cell with surrogate spike gradient."""
    def __init__(self, size: int, tau: float = 0.85, threshold: float = 1.0):
        super().__init__()
        self.size = size
        self.tau = tau
        self.threshold = threshold
        self.register_buffer("mem", torch.empty(0))

    def reset(self, batch_size, device):
        self.mem = torch.zeros(batch_size, self.size, device=device)

    def forward(self, x):
        if self.mem.numel() == 0 or self.mem.shape[0] != x.shape[0]:
            self.reset(x.shape[0], x.device)
        self.mem = self.tau * self.mem + x
        spike = (self.mem >= self.threshold).float()
        # Straight-through surrogate: useful for this small prototype.
        hard = spike.detach()
        soft = torch.sigmoid(10.0 * (self.mem - self.threshold))
        spike = hard + soft - soft.detach()
        self.mem = self.mem * (1.0 - spike.detach())
        return spike
