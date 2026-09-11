# PAC-1 — Persistent Adaptive Cognition

## Research question
Does continuously maintained internal state provide a measurable adaptation advantage over stateless or finite-context inference when an agent operates in a partially observable, non-stationary environment?

## Phase 1
A controlled benchmark with 8 observable contexts, 4 hidden regimes, 4 actions, noisy observations, stochastic regime changes and reward feedback. It compares a stateless MLP, a finite-context Transformer and a persistent GRU, plus an oracle upper-bound reference.

PAC-1 does not test consciousness or prove computational functionalism. It tests the narrower empirical hypothesis that explicitly persistent recurrent state can improve adaptation under repeated distribution shifts.

## Run
```bash
pip install -r requirements.txt
python experiments/sanity_check.py
python experiments/train_and_evaluate.py
```

Results are written to `results/results.json` and model checkpoints to `results/*.pt`.

## Core falsification criterion
If a properly controlled finite-context model performs equivalently to or better than the persistent model on adaptation time, cumulative reward and robustness to longer regime durations, the persistence hypothesis is weakened.


## RadhaKrishna-AI — What It Does and How to Run It
This repo has two parts: a research thesis (theory/spec) and a Phase 1 benchmark implementation (runnable code).

Big Picture
The thesis asks whether machine intelligence can emerge from causal information-processing structure on neuromorphic hardware — not from biological substrate. It combines:

Global Workspace Theory — global broadcasting for conscious access
Predictive Processing — hierarchical error minimization
Spiking Neural Networks — event-driven computation
The long-term vision is a 4-module cognitive architecture (Predictive Hierarchy → Global Workspace Router → Plasticity Engine → Memory Engine) tested on a benchmark called PERCON-Env.

What exists in code today is PAC-1 Phase 1 — a narrower, empirical first step that tests one piece of that vision:

Does explicitly persistent internal state help an agent adapt when the environment keeps changing in hidden ways?

Repository Layout
RadhaKrishna-AI/
├── README.md                          # Top-level thesis overview
├── thesis/
│   ├── Intellectual_Thesis.md         # v1 — philosophy, architecture, falsification
│   └── Intellectual_ThesisV2.md       # v2 — experiments, PERCON-Env spec
└── PAC-1_Phase-1_Implementation/    # Runnable benchmark (this is the source code)
    ├── config.py
    ├── requirements.txt
    ├── environment/hidden_regime.py   # Simulated non-stationary world
    ├── pac1/
    │   ├── models.py                  # 3 agent architectures
    │   ├── data.py                    # Episode generation & dataset building
    │   └── evaluate.py                # Metrics: reward, adaptation time
    └── experiments/
        ├── sanity_check.py            # Quick env smoke test
        └── train_and_evaluate.py      # Full train + compare + save results
How the Code Works (End to End)
1. The Environment — HiddenRegimeEnv

hidden_regime.py
Lines 3-29
class HiddenRegimeEnv:
    def __init__(self, n_contexts=8, n_actions=4, n_regimes=4, obs_noise=0.10, regime_change_prob=0.03, episode_length=500, seed=42):
        ...
        self.mapping=self.rng.integers(0,n_actions,size=(n_regimes,n_contexts))
    ...
    def step(self, action):
        action=int(action); optimal=self.oracle_action(); reward=float(action==optimal)
        ...
        if self.t<self.episode_length and self.rng.random()<self.regime_change_prob:
            candidates=[r for r in range(self.n_regimes) if r!=self.regime]; self.regime=int(self.rng.choice(candidates)); changed=True
        self.context=int(self.rng.integers(0,self.n_contexts))
        ...
        return self._observe(self.context),reward,self.t>=self.episode_length,info
Each step:

Agent sees a noisy one-hot context (8 possible contexts, 10% observation noise).
Agent picks one of 4 actions.
Reward is 1 if correct, 0 if not — but “correct” depends on a hidden regime the agent never observes directly.
Each regime has its own mapping from context → optimal action (a 4×8 lookup table).
With 3% probability per step, the regime switches (distribution shift).
Context also randomizes each step.
So the agent must infer which hidden regime is active and adapt when it changes — classic partially observable, non-stationary learning.

2. Input Representation — pac1/data.py
Each timestep token = [observation (8)] + [previous action one-hot (4)] + [previous reward (1)] → 13 dimensions.

Training uses oracle-labeled episodes: at each step, the target action is whatever the oracle would pick (supervised imitation), not the agent’s own actions. That teaches the models the context→action mapping under each regime before live evaluation.

3. Three Competing Agents — pac1/models.py
Model	Architecture	Memory
StatelessMLP
Linear → ReLU → Linear on current obs only
None
ContextTransformer
2-layer transformer over last 20 tokens
Finite window (20 steps)
PersistentGRU
GRU with carried hidden state
Unlimited (within episode)
This is the core experiment: stateless vs. finite context vs. persistent recurrent state.

4. Training — experiments/train_and_evaluate.py
Flow:

Load hyperparameters from config.py (500 train episodes, 20 epochs, batch 32, lr 3e-4, etc.).
Generate 500 episodes of supervised data.
Train all 3 models with cross-entropy loss (Adam optimizer, gradient clipping).
Evaluate each on 100 fresh episodes.
Save checkpoints to results/*.pt and metrics to results/results.json.
5. Evaluation Metrics — pac1/evaluate.py
For each eval episode, the agent acts autonomously (no oracle labels):

Mean reward — fraction of correct actions
Adaptation time — after each regime change, how many steps until rolling mean reward recovers to 90% of pre-change performance
If the GRU adapts faster and earns more reward than the Transformer and MLP, that supports the persistence hypothesis. If a finite-context model matches or beats the GRU, the hypothesis is weakened.

6. Configuration — config.py
Key defaults:


config.py
Lines 4-22
class Config:
    n_contexts: int = 8
    n_actions: int = 4
    n_regimes: int = 4
    obs_noise: float = 0.10
    regime_change_prob: float = 0.03
    episode_length: int = 500
    train_episodes: int = 500
    eval_episodes: int = 100
    context_window: int = 20
    ...
Steps to Run
Prerequisites
Python 3.10+ (3.12 works; the repo has a .venv with torch already)
Dependencies: numpy, torch
Setup
cd PAC-1_Phase-1_Implementation
# Option A: use existing venv
source .venv/bin/activate
# Option B: fresh setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Step 1 — Sanity check (no PyTorch needed for env only)
python experiments/sanity_check.py
Expected output:

PAC-1 environment sanity check
Episode length: 500
Fixed-action reward: ~111/500
Regime changes: ~14
PASS: environment executes end-to-end.
This confirms the environment runs; a fixed action gets ~22% reward (random would be ~25% with 4 actions, but noise and regime shifts affect it).

Step 2 — Full train + evaluate
python experiments/train_and_evaluate.py
This takes a few minutes on CPU (faster on GPU if available). It prints JSON like:

{
  "config": { ... },
  "device": "cpu",
  "final_training_losses": {
    "stateless": ...,
    "transformer": ...,
    "gru": ...
  },
  "evaluation": {
    "stateless": { "mean_reward": ..., "mean_adaptation_time": ... },
    "transformer": { ... },
    "gru": { ... }
  }
}
Outputs:

results/stateless.pt, results/transformer.pt, results/gru.pt — model weights
results/results.json — full run summary
How This Connects to the Thesis
Thesis concept	PAC-1 Phase 1 implementation
PERCON-Env (non-stationary benchmark)
HiddenRegimeEnv with stochastic regime shifts
Persistent cognition / memory
GRU vs. Transformer vs. MLP comparison
Adaptation latency (τ_adapt)
mean_adaptation_time in evaluate.py
Cognitive Architecture v0.1 (SNN, GWT, STDP)
Not implemented yet — PAC-1 uses standard PyTorch RNN/Transformer baselines
PAC-1 is deliberately scoped: it does not test consciousness or neuromorphic SNNs. It tests whether persistent state gives a measurable edge in a controlled shifting environment — a prerequisite before building the full architecture described in the thesis docs.

Quick Mental Model
┌─────────────────────────────────────────────────────────┐
│              HiddenRegimeEnv (simulated world)           │
│  Hidden: regime (4) → maps context (8) → action (4)     │
│  Observable: noisy context + reward feedback             │
│  Non-stationary: regime switches ~3% per step            │
└──────────────────────┬──────────────────────────────────┘
                       │ obs, reward
                       ▼
        ┌──────────────┴──────────────┐
        │         Agent picks action   │
        └──────────────┬──────────────┘
                       │
     ┌─────────────────┼─────────────────┐
     ▼                 ▼                 ▼
 StatelessMLP   ContextTransformer   PersistentGRU
 (no memory)    (20-step window)     (carried state)
                       │
                       ▼
              Compare: reward & adaptation time
