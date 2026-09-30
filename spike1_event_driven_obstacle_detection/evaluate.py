import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def save_results(results, out_dir="results"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    rows = []
    for name, metrics in results["models"].items():
        row = {"model": name}
        row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)

def plot_activity(results, out_dir="results"):
    out = Path(out_dir)
    names = list(results["models"])
    vals = [
        results["models"][n].get("mean_spikes", 0)
        for n in names
    ]
    plt.figure(figsize=(7, 4))
    plt.bar(names, vals)
    plt.ylabel("Mean spikes / sample")
    plt.title("SNN spike activity")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out / "activity.png", dpi=160)
    plt.close()
