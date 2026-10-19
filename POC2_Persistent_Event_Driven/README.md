# POC2 — Persistent Event-Driven Memory

POC2 tests whether a sparse event-driven model can maintain **task-relevant internal state over long temporal horizons**, instead of repeatedly reconstructing state from the full history.

## Research question

> Can persistent event-driven neural state retain useful information across long event streams while using less computation/context than replaying or retrieving the entire history?

This is a research prototype, not a claim that SNNs solve agent memory.

## Architecture

```text
Event stream
    ↓
Event encoder
    ↓
Persistent SNN
    ├── fast state
    └── slow persistent memory
    ↓
Readout / query
    ↓
Prediction
```

The first benchmark is a delayed-recall task with distractors. A relevant event occurs early, many irrelevant events follow, and a query asks for the remembered value.

## Baselines

- Stateless MLP over the current event
- GRU recurrent baseline
- Persistent LIF SNN

The key comparison is accuracy vs. temporal gap, distractor load, spike activity and an approximate operation count.

## Run

```bash
pip install -r requirements.txt
python -m experiments.train
python -m experiments.memory_sweep
```

Results are written to `results/latest/`.

## Important scientific caveat

The operation count is a **proxy**, not measured energy or hardware execution. A dense PyTorch implementation does not automatically become cheaper merely because its input is sparse. Hardware/sparse-kernel measurements are a later stage.
