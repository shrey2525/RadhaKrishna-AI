import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def accuracy(pred, y):
    return float((pred == y).float().mean().item())
