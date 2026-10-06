import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="results/latest")
args = ap.parse_args()
p = Path(args.run) / "sparsity_results.csv"
if not p.exists():
    raise FileNotFoundError(f"{p} not found. Run experiments/sparsity_sweep.py first.")
df = pd.read_csv(p)
print(df.to_string(index=False))
