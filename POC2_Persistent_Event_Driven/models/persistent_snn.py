import torch
from torch import nn
from .lif import LIFCell

class PersistentSNN(nn.Module):
    """Two-timescale SNN: fast spiking state + slow persistent memory."""
    def __init__(self, input_dim, hidden_dim, memory_dim, output_dim,
                 tau_fast=0.85, tau_slow=0.995):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.fast = LIFCell(hidden_dim, tau=tau_fast)
        self.memory_proj = nn.Linear(hidden_dim, memory_dim)
        self.memory_decay = tau_slow
        self.readout = nn.Linear(hidden_dim + memory_dim, output_dim)
        self.last_spike_count = 0
        self.last_memory_norm = 0.0

    def forward(self, x):
        # x: [batch, time, features]
        self.fast.mem = torch.empty(0, device=x.device)
        memory = torch.zeros(x.shape[0], self.memory_proj.out_features, device=x.device)
        outputs, spike_count = [], 0
        for t in range(x.shape[1]):
            h = self.input_proj(x[:, t])
            spikes = self.fast(h)
            spike_count += int((spikes.detach() > 0).sum().item())
            memory_update = torch.tanh(self.memory_proj(spikes))
            memory = self.memory_decay * memory + (1.0 - self.memory_decay) * memory_update
            outputs.append(self.readout(torch.cat([spikes, memory], dim=-1)))
        self.last_spike_count = spike_count
        self.last_memory_norm = float(memory.detach().norm(dim=-1).mean().item())
        return torch.stack(outputs, dim=1)

    def predict(self, x):
        return self.forward(x)[:, -1].argmax(-1)
