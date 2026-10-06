import argparse, json, random, sys
from pathlib import Path

# Support both `python experiments/train.py` and `python -m experiments.train`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from core.config import load_config
from benchmarks.synthetic import SyntheticTemporalDataset
from models.ann import DenseTemporalClassifier
from models.snn import EventDrivenSNN

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def train(model, loader, epochs, lr, snn=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            out = model(x)
            logits = out.sum(1) if snn else out
            loss_fn(logits, y).backward()
            opt.step()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc1.yaml")
    ap.add_argument("--output", default="results/latest")
    args = ap.parse_args()
    cfg = load_config(args.config); seed_all(cfg["seed"])
    Path(args.output).mkdir(parents=True, exist_ok=True)
    d, m, t = cfg["data"], cfg["model"], cfg["training"]

    ds = SyntheticTemporalDataset(d["train_samples"], d["sequence_length"], d["input_dim"],
                                  d["n_classes"], 1.0, d["noise_std"], cfg["seed"])
    loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=True)

    ann = DenseTemporalClassifier(d["sequence_length"], d["input_dim"], m["hidden_dim"], d["n_classes"])
    snn = EventDrivenSNN(d["input_dim"], m["hidden_dim"], d["n_classes"])
    train(ann, loader, t["epochs"], t["learning_rate"], False)
    train(snn, loader, t["epochs"], t["learning_rate"], True)

    torch.save(ann.state_dict(), Path(args.output) / "ann.pt")
    torch.save(snn.state_dict(), Path(args.output) / "snn.pt")
    meta = {"config": cfg,
            "ann_parameters": sum(p.numel() for p in ann.parameters()),
            "snn_parameters": sum(p.numel() for p in snn.parameters())}
    (Path(args.output) / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
