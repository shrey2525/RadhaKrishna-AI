# POC1 — Event-Driven Intelligence

A domain-agnostic benchmark for testing whether sparse, event-driven SNN inference can reduce computation while preserving task performance.

Research question: **At what event sparsity does event-driven SNN inference become more computationally efficient than a comparable dense neural network?**

Architecture:
Application → Event Adapter → Event Encoder → ANN/SNN → Decision → Metrics

POC1 intentionally does not test persistence, self-models, consciousness, or hardware energy. It establishes the compute-efficiency foundation.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python experiments/train.py
python experiments/sparsity_sweep.py
python experiments/evaluate.py --run results/latest
streamlit run visualization/dashboard.py
```

The included synthetic benchmark varies temporal event density and reports accuracy, F1, spike counts, operation proxies, latency, and compute-reduction proxy.

**Important:** operation counts are simulation proxies, not physical energy measurements. Hardware energy claims require hardware measurement.
