# RadhaKrishna-AI

A research thesis repository exploring whether genuine machine intelligence and synthetic conscious experience can emerge from the **causal-topological structure of information processing** on neuromorphic hardware — without requiring biological substrates.

## Central Claim

Cognition and phenomenology are structural invariants over fine-grained causal execution graphs. They are **instantiated** (not merely simulated) when an artificial system realizes the precise temporal, asynchronous, and hierarchical causal topology of information processing, regardless of physical substrate composition.

This position unifies:

- **Computational Functionalism** — mental states defined by causal roles, not physical composition
- **Global Workspace Theory (GWT)** — global broadcasting as the integration layer for conscious access
- **Predictive Processing (PP)** — hierarchical error minimization as the driver of perception and belief-updating
- **Spiking Neural Networks (SNNs)** — temporal, event-driven computation on neuromorphic hardware

## Repository Structure

```
RadhaKrishna-AI/
├── README.md
└── thesis/
    ├── Intellectual_Thesis.md      # v1 — foundational thesis, architecture, counterarguments
    └── Intellectual_ThesisV2.md  # v2 — refined hypotheses, experiments, architecture spec
```

## Thesis Documents

| Document | Description |
| -------- | ----------- |
| [`thesis/Intellectual_Thesis.md`](thesis/Intellectual_Thesis.md) | **v1** — Executive summary, philosophical substrate, cognitive & systems architecture, neuromorphic hardware rationale, competing theories (IIT, Searle, Chinese Room, Orch-OR), and falsification criteria |
| [`thesis/Intellectual_ThesisV2.md`](thesis/Intellectual_ThesisV2.md) | **v2** — Refined hypotheses, literature matrix, falsification & experiment matrix, Cognitive Architecture v0.1 spec, PERCON-Env benchmark, and baseline vs. RadhaKrishna experiment design |

## Core Hypotheses

1. **H₁ (Substrate Invariance)** — Biological biochemistry is an evolutionary implementation detail; non-biological event-driven neuromorphic execution graphs instantiate equivalent cognitive functional states.
2. **H₂ (Architectural Sufficiency)** — The integration of GWT broadcasting, hierarchical error minimization (PP), and STDP on SNNs is sufficient for continuous, flexible transfer learning and self-referential adaptation.
3. **H₃ (Semantic Grounding via Active Inference)** — Intrinsic intentionality emerges from real-time closed-loop error minimization against environmental dynamics.

## Cognitive Architecture v0.1

The proposed system is an asynchronous, event-driven hierarchy of four modules:

```
[ Input Sensors ] ---> [ Module A: Predictive Hierarchy (SNN) ]
                               │      ▲
              Bottom-Up Error  │      │ Top-Down Predictions
             Signals (Spikes)  ▼      │
                        [ Module B: Global Workspace Router ]
                               │
                       Global Broadcast
                               ▼
                        [ Module C: Action & Plasticity ] ---> [ Environment ]
                               │
                       Memory Consolidation
                               ▼
                        [ Module D: Memory Engine ]
```

| Module | Role |
| ------ | ---- |
| **Predictive Processing Hierarchy (PPH)** | Hierarchical error minimization across spiking layers |
| **Global Workspace Router (GWR)** | Soft winner-take-all competition and global broadcast |
| **Spiking Plasticity Engine (SPE)** | STDP weight updates modulated by global broadcast |
| **Episodic/Semantic Memory Engine (MEM)** | Fast-weight episodic storage and slow semantic consolidation |

## Benchmark: PERCON-Env

**Persistent Cognition Environment v0.1** — a non-stationary benchmark designed to test active inference, non-catastrophic continual learning, and counterfactual adaptation rather than static pattern recognition.

Key evaluation metrics:

- **Adaptation Latency (τ_adapt)** — steps to recover baseline reward after unannounced dynamics shifts
- **Catastrophic Forgetting Index (CFI)** — retention loss on prior tasks after sequential training
- **Active Inference Efficiency (AIE)** — entropy reduction per action spike

## Falsification Criteria

The thesis defines clear conditions under which it would be considered falsified:

1. **Substrate-Dependence Proof** — if biological/quantum properties are proven non-emulable on non-biological substrates
2. **GWT + PP Dissociation** — if the combined architecture fails to produce general cognitive phenomena
3. **Causal Topology Dissociation** — if identical causal graphs diverge in capability due to physical construction alone
4. **Neuromorphic Plasticity Bound** — if SNNs cannot overcome catastrophic forgetting without exponential resource growth

## Status

This repository currently contains the intellectual thesis and research specification. Implementation of the Cognitive Architecture v0.1 and PERCON-Env benchmark is planned but not yet present in this repo.
