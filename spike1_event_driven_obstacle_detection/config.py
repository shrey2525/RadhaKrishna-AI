from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    samples: int = 2000
    test_fraction: float = 0.20

    width: int = 32
    height: int = 32
    duration_ms: int = 100
    time_bins: int = 20

    object_size: int = 5
    background_events: int = 3

    ann_hidden: int = 64
    ann_epochs: int = 15
    ann_lr: float = 1e-3

    snn_hidden: int = 128
    snn_threshold: float = 1.0
    snn_tau_ms: float = 10.0
    snn_tau_out_ms: float = 20.0

    batch_size: int = 64
