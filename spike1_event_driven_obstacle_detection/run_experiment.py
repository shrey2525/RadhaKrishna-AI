import argparse
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config import Config
from generate_data import generate_dataset
from baseline_ann import train_and_evaluate
from snn_brian2 import train_snn, evaluate_snn
from evaluate import save_results, plot_activity

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--duration-ms", type=int, default=100)
    args = p.parse_args()

    cfg = Config(
        seed=args.seed,
        samples=args.samples,
        ann_epochs=args.epochs,
        duration_ms=args.duration_ms,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print("=" * 68)
    print("SPIKE 1 — Event-driven obstacle detection")
    print("=" * 68)

    samples, labels = generate_dataset(cfg)
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=cfg.test_fraction,
        random_state=cfg.seed,
        stratify=labels,
    )

    print(f"Samples: {len(labels)} | Train: {len(train_idx)} | Test: {len(test_idx)}")

    print("\n[1/2] Training ANN baseline...")
    ann = train_and_evaluate(samples, labels, train_idx, test_idx, cfg)

    print("[2/2] Running Brian2 SNN...")
    weights = train_snn(samples, labels, train_idx, cfg)
    snn = evaluate_snn(samples, labels, test_idx, cfg, weights)

    results = {
        "experiment": "Spike 1 - synthetic event obstacle detection",
        "hypothesis": "Direct sparse-event processing can retain fidelity while reducing active computation.",
        "config": vars(cfg),
        "models": {
            "ann_event_histogram": ann,
            "brian2_snn": snn,
        },
        "targets": {
            "fidelity_fraction_of_ann": 0.99,
            "active_computation_reduction": "10x target",
            "note": "Targets are hypotheses, not achieved results. CPU simulation is not hardware energy measurement.",
        },
    }

    save_results(results)
    plot_activity(results)

    print("\nRESULTS")
    print("-" * 68)
    print(f"ANN accuracy:       {ann['accuracy']:.4f}")
    print(f"SNN accuracy:       {snn['accuracy']:.4f}")
    print(f"ANN F1:             {ann['f1']:.4f}")
    print(f"SNN F1:             {snn['f1']:.4f}")
    print(f"SNN mean latency:   {snn['mean_decision_latency_ms']:.3f} ms")
    print(f"SNN mean spikes:    {snn['mean_spikes']:.1f}")
    print(f"Input events/sample:{snn['mean_input_events']:.1f}")
    print("\nSaved under results/")

if __name__ == "__main__":
    main()
