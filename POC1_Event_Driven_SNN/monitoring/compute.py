import time
from dataclasses import dataclass, asdict

@dataclass
class Metrics:
    latency_ms: float
    input_events: int
    output_spikes: int
    synops_proxy: int

    def to_dict(self):
        return asdict(self)

def ann_macs(sequence_length, input_dim, hidden_dim, output_dim, samples):
    per_sample = sequence_length * input_dim * hidden_dim + hidden_dim * output_dim
    return per_sample * samples

def benchmark_snn(model, x):
    start = time.perf_counter()
    with __import__("torch").no_grad():
        spikes = model(x)
    latency = (time.perf_counter() - start) * 1000
    input_events = int((x > 0).sum().item())
    output_spikes = int(spikes.sum().item())
    # Conservative simulation-level proxy:
    # input spikes drive fc1; output spikes drive fc2.
    synops = input_events * model.fc1.weight.shape[0] + output_spikes * model.fc2.weight.shape[1]
    return Metrics(latency, input_events, output_spikes, synops)
