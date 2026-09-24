import numpy as np

def _add_object_events(events, x, y, polarity, t, size, width, height, rng):
    for _ in range(size * size):
        dx = int(rng.integers(0, size))
        dy = int(rng.integers(0, size))
        xx = int(np.clip(x + dx + rng.integers(-1, 2), 0, width - 1))
        yy = int(np.clip(y + dy + rng.integers(-1, 2), 0, height - 1))
        events.append((xx, yy, t, polarity))

def generate_sample(cfg, label, rng):
    events = []
    x0 = 2
    y0 = cfg.height // 2 - cfg.object_size // 2
    span = cfg.width - cfg.object_size - 4

    for step in range(cfg.time_bins):
        frac = step / max(cfg.time_bins - 1, 1)
        x = int(round(x0 + span * (frac if label == 0 else 1.0 - frac)))
        polarity = 1 if label == 0 else -1
        t0 = step * cfg.duration_ms / cfg.time_bins
        t = (step + 0.5) * cfg.duration_ms / cfg.time_bins

        _add_object_events(
            events, x, y0, polarity, t, cfg.object_size,
            cfg.width, cfg.height, rng
        )

        for _ in range(cfg.background_events):
            events.append((
                int(rng.integers(0, cfg.width)),
                int(rng.integers(0, cfg.height)),
                float(rng.uniform(t0, t0 + cfg.duration_ms / cfg.time_bins)),
                int(rng.choice([-1, 1])),
            ))

    return np.asarray(events, dtype=np.float32)

def generate_dataset(cfg):
    rng = np.random.default_rng(cfg.seed)
    samples, labels = [], []
    for i in range(cfg.samples):
        labels.append(i % 2)
        samples.append(generate_sample(cfg, i % 2, rng))
    return samples, np.asarray(labels, dtype=np.int64)

def event_histogram(events, cfg):
    hist = np.zeros((2, cfg.height, cfg.width), dtype=np.float32)
    for x, y, _, p in events:
        hist[0 if p < 0 else 1, int(y), int(x)] += 1.0
    return hist / max(float(hist.max()), 1.0)
