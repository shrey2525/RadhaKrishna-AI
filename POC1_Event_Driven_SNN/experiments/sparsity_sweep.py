import argparse, json, sys
from pathlib import Path

# Support both `python experiments/sparsity_sweep.py` and module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from core.config import load_config
from benchmarks.synthetic import SyntheticTemporalDataset
from models.ann import DenseTemporalClassifier
from models.snn import EventDrivenSNN
from monitoring.compute import ann_macs, benchmark_snn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc1.yaml")
    ap.add_argument("--run", default="results/latest")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    run = Path(args.run)

    ann = DenseTemporalClassifier(d["sequence_length"], d["input_dim"], m["hidden_dim"], d["n_classes"])
    snn = EventDrivenSNN(d["input_dim"], m["hidden_dim"], d["n_classes"])
    ann.load_state_dict(torch.load(run / "ann.pt", map_location="cpu"))
    snn.load_state_dict(torch.load(run / "snn.pt", map_location="cpu"))

    rows = []
    for density in cfg["benchmark"]["event_densities"]:
        ds = SyntheticTemporalDataset(d["test_samples"], d["sequence_length"], d["input_dim"],
                                      d["n_classes"], density, d["noise_std"],
                                      cfg["seed"] + round(density * 1000))
        loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=False)
        ann_preds, snn_preds, labels = [], [], []
        total = {"latency_ms": 0, "input_events": 0, "output_spikes": 0, "synops_proxy": 0}

        with torch.no_grad():
            for x, y in loader:
                ann_preds += ann(x).argmax(-1).tolist()
                metric = benchmark_snn(snn, x)
                snn_preds += snn.predict(x).tolist()
                labels += y.tolist()
                for k in total: total[k] += getattr(metric, k)

        ann_compute = ann_macs(d["sequence_length"], d["input_dim"], m["hidden_dim"],
                               d["n_classes"], len(ds))
        reduction = 1 - total["synops_proxy"] / max(ann_compute, 1)

        rows.append({
            "event_density": density,
            "ann_accuracy": accuracy_score(labels, ann_preds),
            "snn_accuracy": accuracy_score(labels, snn_preds),
            "ann_f1": f1_score(labels, ann_preds, average="macro"),
            "snn_f1": f1_score(labels, snn_preds, average="macro"),
            "ann_macs_proxy": ann_compute,
            "snn_synops_proxy": total["synops_proxy"],
            "compute_reduction_proxy": reduction,
            "efficiency_ratio_proxy": ann_compute / max(total["synops_proxy"], 1),
            "snn_latency_ms": total["latency_ms"],
            "input_events": total["input_events"],
            "output_spikes": total["output_spikes"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(run / "sparsity_results.csv", index=False)
    (run / "sparsity_results.json").write_text(json.dumps(rows, indent=2))
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
