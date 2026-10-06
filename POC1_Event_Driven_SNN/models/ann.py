import torch
from torch import nn

class DenseTemporalClassifier(nn.Module):
    def __init__(self, sequence_length, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sequence_length * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x.flatten(1))
