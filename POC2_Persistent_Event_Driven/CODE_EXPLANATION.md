# POC2 Code Explanation Guide

This guide explains how the code in `POC2_Persistent_Event_Driven` is organized and how its delayed-recall experiment works. POC2 is a research prototype for comparing a recurrent baseline with a spiking model that keeps a slowly changing state while processing a sequence.

## What the experiment asks

Each example contains a class cue near the start of a sequence, random distractor events, and a query marker later in the sequence. A model must predict the class of the earlier cue. The experiment varies the distance between the cue and the query to examine how performance changes as the recall delay grows.

The main comparison is between:

- **GRU:** processes the sequence recurrently and classifies from its final hidden state.
- **Persistent SNN:** processes each time step with a leaky integrate-and-fire (LIF) layer and updates a slower memory vector.
- **Stateless MLP:** implemented as a simple model, but not currently included in the training or memory-sweep scripts.

“Persistent” here means that the memory vector changes slowly across time steps *inside one input sequence*. Both the fast LIF state and slow memory are reset when a new sequence is passed to the model; the state is not saved between batches or runs.

## Project map

| Path | Purpose |
| --- | --- |
| `configs/poc2.yaml` | Dataset, model, training, and sweep settings. |
| `benchmarks/delayed_recall.py` | Builds the synthetic delayed-recall dataset. |
| `models/gru.py` | GRU sequence classifier baseline. |
| `models/lif.py` | LIF neuron cell and its surrogate gradient. |
| `models/persistent_snn.py` | Fast spiking state, slow memory update, and output layer. |
| `models/stateless.py` | Stateless MLP model; currently not used by the experiment scripts. |
| `experiments/common.py` | Seed setup and a small accuracy helper. |
| `experiments/train.py` | Trains the GRU and Persistent SNN and saves their weights. |
| `experiments/memory_sweep.py` | Measures accuracy at several recall gaps and writes CSV/JSON output. |
| `core/event.py` | General-purpose event data structure; not currently used by the benchmark pipeline. |
| `core/memory.py` | Standalone decaying state helper; not used by `PersistentSNN`. |
| `monitoring/compute.py` | Dense-operation and active-synapse proxy calculations; not currently called by the experiment scripts. |
| `tests/test_memory.py` | Small checks for the standalone persistent state helper and LIF cell. |

The `__init__.py` files make the directories importable as Python packages. `models/__init__.py` also re-exports the model classes so scripts can import them from `models`.

## Data generation

`DelayedRecallDataset` returns one pair `(x, y)` per example:

- `x` has shape `[sequence_length, input_dim]` for one example.
- `y` is the class index to predict.
- A batch from the data loader has `x` shaped `[batch, time, input_dim]` and `y` shaped `[batch]`.

The generator starts with an all-zero tensor, then:

1. Places a one-hot class cue at time step `0` using the class index as the feature index.
2. Adds random one-feature distractors at intermediate time steps according to `distractor_probability`.
3. Places the query marker in the final feature channel at `min(sequence_length - 1, memory_gap + 1)`.

The configured training task uses 16 input features, 4 classes, a sequence length of 100, a gap of 50, and a distractor probability of 0.15. The benchmark gap sweep uses `[10, 25, 50, 100, 250, 500]`. For each gap, the sweep makes the sequence at least `gap + 2` steps long, so the query can be placed after that delay.

The query feature is a marker in the input. It does not select a separate readout operation: the scripts classify from the model’s output at the final sequence step. For gaps shorter than the configured base sequence, the query can occur before the end and the model still has to produce its final prediction at the end.

## Model details

### GRU baseline

`GRUClassifier` passes the full sequence through `torch.nn.GRU`. With `batch_first=True`, it expects `[batch, time, features]`. It uses the final hidden state, shaped `[batch, hidden_dim]`, and a linear layer to produce class logits shaped `[batch, n_classes]`.

### LIF cell

`LIFCell` maintains a membrane value for each item in the batch and hidden unit. At each time step it applies:

```text
membrane = tau * membrane + input
```

When the membrane reaches the threshold, the cell emits a spike and resets that unit’s membrane. The forward pass uses a hard threshold, while the backward pass uses a sigmoid-based straight-through surrogate so gradients can flow through this discrete spike decision during training.

### Persistent SNN

`PersistentSNN` performs these operations for every time step:

1. Projects the input features into the hidden dimension.
2. Sends that projection through the fast LIF cell to produce spikes.
3. Projects the spikes into the memory dimension and applies `tanh`.
4. Updates the slow memory using an exponential blend:

   ```text
   memory = tau_slow * memory + (1 - tau_slow) * memory_update
   ```

5. Concatenates the current spikes and memory and maps them to class logits.

The result is a sequence of logits shaped `[batch, time, n_classes]`. Training and evaluation select the final time step. The model also records a spike count and final memory norm for inspection; the sweep currently reports the spike count but not the memory norm.

The slow memory is initialized to zero for each `forward` call, while `tau_slow=0.995` retains most of the previous value at each step. This gives a longer time scale than the fast LIF state. It is a simple learned-feature accumulator, not an external store or semantic memory system.

## Training flow

`experiments/train.py` does the following:

1. Loads `configs/poc2.yaml` and seeds Python, NumPy, and PyTorch.
2. Generates the configured training set and wraps it in a shuffled `DataLoader`.
3. Creates a GRU and a Persistent SNN using the configured dimensions and time constants.
4. Trains each model with Adam and cross-entropy loss. For the SNN, it selects the final time step’s logits; the GRU already returns one set of logits per sequence.
5. Prints mean training loss per epoch and saves each model’s `state_dict` under `results/latest/`.

The script currently reports training loss only. It does not make a validation set or report held-out accuracy during training. The models use PyTorch’s default device, which is CPU unless the code is changed to move them and the input batches to another device.

## Memory-gap sweep

`experiments/memory_sweep.py` creates a new test set for each configured gap, then evaluates a GRU and a Persistent SNN. When matching checkpoint files exist in `results/latest/`, it loads them; otherwise, the model remains randomly initialized. Therefore, run training first if the goal is to measure the trained models. Without checkpoints, the sweep is a random-weight diagnostic and its accuracy should not be interpreted as trained performance.

For each model and gap, the script records accuracy. It also records `last_spike_count` for the SNN; the GRU has no such field, so its recorded value is zero. The sweep writes:

- `results/latest/memory_sweep.csv`
- `results/latest/memory_sweep.json`

The `is_snn` argument in its `evaluate` helper is currently unused. The sweep does not call the operation-count helpers in `monitoring/compute.py`.

## Configuration reference

The main settings in `configs/poc2.yaml` are:

| Setting | What it controls |
| --- | --- |
| `seed` | Initial random seed for dataset generation and training. |
| `data.input_dim`, `data.n_classes` | Input feature count and number of target classes. |
| `data.sequence_length`, `data.memory_gap` | Training sequence length and cue-to-query delay. |
| `data.distractor_probability` | Chance of adding a random distractor at an eligible time step. |
| `data.train_samples`, `data.test_samples` | Number of generated training and sweep examples. |
| `model.hidden_dim`, `model.memory_dim` | Fast hidden-state size and slow-memory size. |
| `model.tau_fast`, `model.tau_slow` | LIF membrane retention and slow-memory retention. |
| `training.epochs`, `training.batch_size`, `training.learning_rate` | Training loop settings. |
| `benchmark.gaps` | Recall delays evaluated by the memory sweep. |

## Running the experiment

Run the commands from the POC2 project directory so the package imports and the memory-sweep script’s relative config/result paths resolve correctly:

```bash
cd POC2_Persistent_Event_Driven
pip install -r requirements.txt
python -m experiments.train
python -m experiments.memory_sweep
```

Both experiment entry points also support direct execution after their import-path fixes:

```bash
python experiments/train.py
python experiments/memory_sweep.py
```

Both scripts resolve imports, config, checkpoints, and results relative to the POC2 directory, so direct execution does not depend on the current working directory.

## Reading the results carefully

This prototype compares a compact synthetic task. It does not establish that the SNN is more efficient on real hardware. `monitoring/compute.py` contains theoretical operation proxies, and its own comments state that these are not measured hardware costs. The SNN uses dense PyTorch linear layers, so sparse spikes alone do not imply lower runtime or energy. The sweep also varies sequence length for longer gaps and reports final-step classification; those details matter when interpreting accuracy changes.
