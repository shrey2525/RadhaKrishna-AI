import numpy as np
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, SpikeGeneratorGroup,
    Network, ms, prefs, start_scope
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

prefs.codegen.target = "numpy"

def build_weights(cfg):
    rng = np.random.default_rng(cfg.seed)
    n_in = 2 * cfg.width * cfg.height
    w_in = rng.uniform(0.0, 0.08, size=(n_in, cfg.snn_hidden))
    w_out = rng.uniform(0.0, 0.1, size=(cfg.snn_hidden, 2))
    return {"in": w_in, "out": w_out}

def train_snn(samples, labels, train_idx, cfg):
    # Transparent first-spike training: use a deterministic temporal reservoir
    # and class prototypes to initialize the output readout.
    weights = build_weights(cfg)
    hidden = cfg.snn_hidden
    activity = np.zeros((len(train_idx), hidden), dtype=np.float32)

    for row, idx in enumerate(train_idx):
        for x, y, t, p in samples[idx]:
            neuron = (
                int(y) * cfg.width + int(x)
            )
            neuron = (
                neuron * 131 + int(t * 10) + (0 if p < 0 else 17)
            ) % hidden
            activity[row, neuron] += 1

    for c in range(2):
        mask = labels[train_idx] == c
        centroid = activity[mask].mean(0)
        weights["out"][:, c] = 0.8 * centroid / (centroid.max() + 1e-6)

    return weights

def run_sample(events, cfg, weights, seed):
    start_scope()
    np.random.seed(seed)

    n_in = 2 * cfg.width * cfg.height

    # ------------------------------------------------------------
    # Convert event stream into Brian2 spike-generator input
    # ------------------------------------------------------------
    indices = []
    times = []

    for x, y, t, p in events:
        base = 0 if p < 0 else cfg.width * cfg.height

        neuron_index = (
            base
            + int(y) * cfg.width
            + int(x)
        )

        indices.append(neuron_index)
        times.append(float(t))

    if len(times) == 0:
        indices = np.array([], dtype=int)
        times = np.array([], dtype=float)
    else:
        indices = np.asarray(indices, dtype=int)
        times = np.asarray(times, dtype=float)

        # Brian2 uses a 100 microsecond default timestep.
        # Multiple spikes from the same neuron inside one timestep
        # are not permitted. We therefore quantise events to the
        # simulation timestep and keep at most one spike per neuron
        # per timestep.
        dt_ms = 0.1

        time_steps = np.round(times / dt_ms).astype(np.int64)

        # Keep one event for each (neuron, timestep) pair.
        unique_keys = {}
        for neuron, step, original_time in zip(
            indices,
            time_steps,
            times,
        ):
            key = (int(neuron), int(step))

            if key not in unique_keys:
                unique_keys[key] = float(step * dt_ms)

        indices = np.asarray(
            [key[0] for key in unique_keys],
            dtype=int,
        )

        times = np.asarray(
            [time for time in unique_keys.values()],
            dtype=float,
        )

        # Sort chronologically.
        order = np.argsort(times)
        indices = indices[order]
        times = times[order]

    # ------------------------------------------------------------
    # Input population
    # ------------------------------------------------------------
    inp = SpikeGeneratorGroup(
        N=n_in,
        indices=indices,
        times=times * ms,
        sorted=True,
    )

    # ------------------------------------------------------------
    # Hidden LIF population
    #
    # IMPORTANT:
    # The threshold is written directly into the expression.
    # There is deliberately NO external "threshold" variable.
    # ------------------------------------------------------------
    hidden = NeuronGroup(
        N=cfg.snn_hidden,
        model="""
            dv/dt = -v / tau : 1
        """,
        threshold=f"v > {float(cfg.snn_threshold)}",
        reset="v = 0",
        refractory=2 * ms,
        method="exact",
        namespace={
            "tau": cfg.snn_tau_ms * ms,
        },
    )

    # ------------------------------------------------------------
    # Output population
    #
    # Again, threshold is a literal number rather than an
    # external Brian2 namespace variable.
    # ------------------------------------------------------------
    output = NeuronGroup(
        N=2,
        model="""
            dv/dt = -v / tau_out : 1
        """,
        threshold="v > 1",
        reset="v = 0",
        refractory=2 * ms,
        method="exact",
        namespace={
            "tau_out": cfg.snn_tau_out_ms * ms,
        },
    )

    # ------------------------------------------------------------
    # Input -> hidden synapses
    # ------------------------------------------------------------
    syn_in = Synapses(
        inp,
        hidden,
        model="w : 1",
        on_pre="v_post += w",
    )

    syn_in.connect()

    # Brian2 stores one weight per synapse.
    # Flatten the [input, hidden] matrix.
    syn_in.w = weights["in"].ravel()

    # ------------------------------------------------------------
    # Hidden -> output synapses
    # ------------------------------------------------------------
    syn_out = Synapses(
        hidden,
        output,
        model="w : 1",
        on_pre="v_post += w",
    )

    syn_out.connect()

    syn_out.w = weights["out"].ravel()

    # ------------------------------------------------------------
    # Monitors
    # ------------------------------------------------------------
    hidden_spikes = SpikeMonitor(hidden)
    output_spikes = SpikeMonitor(output)

    # ------------------------------------------------------------
    # Network
    # ------------------------------------------------------------
    net = Network(
        inp,
        hidden,
        output,
        syn_in,
        syn_out,
        hidden_spikes,
        output_spikes,
    )

    # ------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------
    net.run((cfg.duration_ms + 5) * ms)

    # ------------------------------------------------------------
    # Read output
    # ------------------------------------------------------------
    output_counts = np.asarray(
        output_spikes.count[:],
        dtype=int,
    )

    if output_counts.sum() > 0:
        prediction = int(np.argmax(output_counts))
    else:
        prediction = 0

    # First output spike = decision latency.
    if len(output_spikes.t) > 0:
        latency_ms = float(
            np.asarray(output_spikes.t / ms)[0]
        )
    else:
        latency_ms = float(cfg.duration_ms)

    return {
        "prediction": prediction,
        "latency_ms": latency_ms,
        "hidden_spikes": int(
            np.sum(hidden_spikes.count[:])
        ),
        "output_spikes": int(
            np.sum(output_spikes.count[:])
        ),
        "input_events": int(len(events)),
    }

def evaluate_snn(samples, labels, test_idx, cfg, weights):
    results = [
        run_sample(samples[idx], cfg, weights, cfg.seed + i)
        for i, idx in enumerate(test_idx)
    ]

    preds = [r["prediction"] for r in results]
    truth = labels[test_idx]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, preds, average="binary", zero_division=0
    )

    spikes = [
        r["hidden_spikes"] + r["output_spikes"] for r in results
    ]
    events = [r["input_events"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    return {
        "accuracy": float(accuracy_score(truth, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_decision_latency_ms": float(np.mean(latencies)),
        "median_decision_latency_ms": float(np.median(latencies)),
        "mean_input_events": float(np.mean(events)),
        "mean_spikes": float(np.mean(spikes)),
        "spikes_per_input_event": float(np.mean(spikes) / max(np.mean(events), 1)),
        "activity_proxy": float(np.mean(spikes) / max(cfg.snn_hidden * cfg.duration_ms, 1)),
    }
