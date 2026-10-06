import torch
from torch import nn
from spikingjelly.activation_based import neuron, functional, layer

class EventDrivenSNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = layer.Linear(input_dim, hidden_dim)
        self.lif1 = neuron.LIFNode()
        self.fc2 = layer.Linear(hidden_dim, output_dim)
        self.lif2 = neuron.LIFNode()

    def forward(self, x):
        functional.reset_net(self)
        outputs = []
        for t in range(x.shape[1]):
            h = self.lif1(self.fc1(x[:, t]))
            y = self.lif2(self.fc2(h))
            outputs.append(y)
        return torch.stack(outputs, dim=1)

    def predict(self, x):
        return self.forward(x).sum(1).argmax(-1)
