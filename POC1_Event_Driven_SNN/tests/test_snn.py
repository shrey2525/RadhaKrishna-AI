import torch
from models.snn import EventDrivenSNN

def test_snn_forward():
    model = EventDrivenSNN(8, 16, 3)
    out = model(torch.rand(4, 10, 8))
    assert out.shape == (4, 10, 3)
