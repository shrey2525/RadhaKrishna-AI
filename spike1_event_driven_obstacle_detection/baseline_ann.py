import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from generate_data import event_histogram

class SmallANN(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)

def train_and_evaluate(samples, labels, train_idx, test_idx, cfg):
    torch.manual_seed(cfg.seed)
    X = np.stack([event_histogram(e, cfg) for e in samples])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X[train_idx], y[train_idx]),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    model = SmallANN(2 * cfg.height * cfg.width, cfg.ann_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ann_lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(cfg.ann_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(X[test_idx]).argmax(1).numpy()

    truth = labels[test_idx]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, pred, average="binary", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dense_input_values_per_sample": int(2 * cfg.width * cfg.height),
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
    }
