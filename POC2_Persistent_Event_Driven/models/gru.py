from torch import nn

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])
