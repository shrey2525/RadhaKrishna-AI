from dataclasses import dataclass

@dataclass
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
    hidden_size: int = 64
    transformer_layers: int = 2
    transformer_heads: int = 4
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 3e-4
    adaptation_window: int = 20
    recovery_fraction: float = 0.90
    seed: int = 42
