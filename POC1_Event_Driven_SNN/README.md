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

## Optional: capture local Codex telemetry

The workspace-level `.codex/hooks.json` configures Codex to record session and tool lifecycle events in `results/live/codex_events.jsonl`. The listener stores event types, coarse tool categories, hashed identifiers, explicit outcome flags when available, and payload sizes. It does not store prompts, commands, tool output, transcript paths, or working-directory paths.

Project hooks require review and trust in Codex before they run. After reviewing the hook with `/hooks` and restarting or opening a Codex session in this repository, use Codex normally. See the [official Codex Hooks documentation](https://learn.chatgpt.com/docs/hooks) for hook lifecycle and trust behavior. Then summarize captured events with:

```bash
python experiments/report_codex_telemetry.py
```

The existing ANN/SNN weights were trained on the synthetic classification task and are not trained to classify Codex events. The telemetry adapter emits 32-value event vectors as a foundation for a separate, labeled telemetry experiment; the current collector and report do not claim telemetry-model accuracy.

The hook schema and privacy filtering can be exercised offline with representative Codex hook payloads:

```bash
python -m unittest discover -s tests -p 'test_codex_telemetry.py'
```
