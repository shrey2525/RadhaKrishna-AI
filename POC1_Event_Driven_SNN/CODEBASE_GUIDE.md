# POC1 Event-Driven SNN: Codebase Guide

This guide explains what the POC does, how its folders fit together, what the run commands create, and how to read the results. Run the commands below from the `POC1_Event_Driven_SNN` folder.

## What this POC is testing

The project compares two classifiers on the same synthetic time-series task:

- A dense artificial neural network (ANN), which processes the whole sequence with ordinary linear layers.
- A spiking neural network (SNN), which processes one time step at a time and emits spikes through leaky integrate-and-fire neurons.

The experiment reduces the input event density at evaluation time and records how accuracy and a rough operation-count estimate change. It is a software simulation. It does **not** measure hardware energy or prove that the SNN runs faster on a particular device.

## How data flows

```text
configs/poc1.yaml
        |
        v
synthetic labeled sequences --> train ANN and SNN --> save model weights
                                                         |
                                                         v
                                      evaluate each configured event density
                                                         |
                                                         v
                                  CSV/JSON results --> console and dashboard
```

The current train-and-evaluate scripts use `SyntheticTemporalDataset` directly. The separate event adapter, delta encoder, and hardware backend files are extension points; they are not part of this experiment's active path.

## Folder and file guide

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview and quick-start commands. |
| `requirements.txt` | Python dependencies for training, SNN layers, metrics, plotting, and the dashboard. |
| `configs/poc1.yaml` | Dataset sizes, sequence shape, model size, training settings, random seed, and event densities for the sweep. |
| `core/config.py` | Reads a YAML configuration file into a Python dictionary. |
| `core/event.py` | Defines a generic event record with timestamp, source, type, feature data, and optional metadata. |
| `adapters/generic.py` | Generic adapter that currently passes an iterable of events through as a list. |
| `encoders/delta.py` | Turns changes between adjacent time steps into binary events using a threshold. It is not called by the current experiment scripts. |
| `benchmarks/synthetic.py` | Builds labeled, synthetic temporal sequences and applies an event-density mask. |
| `models/ann.py` | Dense baseline: flatten the sequence, then run two linear layers with ReLU between them. |
| `models/snn.py` | SNN: at each time step, run input and output linear layers with leaky integrate-and-fire neurons. Resets neuron state at the start of each forward pass. |
| `experiments/train.py` | Creates the training dataset, trains both models, and saves their weights plus metadata. |
| `monitoring/compute.py` | Computes the ANN MAC estimate and SNN spike, latency, and synaptic-operation proxies. |
| `experiments/sparsity_sweep.py` | Loads the trained weights, tests all configured event densities, and saves the metric table. |
| `experiments/evaluate.py` | Reads and prints the saved CSV; it does not run model inference itself. |
| `visualization/dashboard.py` | Displays charts and the results table using Streamlit. |
| `hardware/backend.py` | Abstract interface for a future hardware backend; it contains no concrete hardware implementation. |
| `tests/` | Small checks for dataset shape, delta encoding, and SNN output shape. |
| `results/latest/` | Default location for trained weights, metadata, and sweep results. |

## Synthetic task and configuration

The default config has 3 classes. Each sample is a sequence with 50 time steps and 32 input features. A class label selects a simple signal pattern: a positive burst in a class-specific time window and feature range. The code adds small random noise, then applies a random mask. `event_density: 1.0` keeps every input position; `0.10` keeps about one in ten positions on average; `0.01` keeps about one in a hundred.

The defaults in `configs/poc1.yaml` are:

- 6,000 training samples and 2,000 test samples per density.
- 128 hidden units in each model.
- 12 training epochs, batches of 128, and learning rate 0.001.
- Evaluation densities of 1.0, 0.5, 0.25, 0.10, 0.05, and 0.01.
- Seed 42, used to make data generation and model initialization repeatable in typical runs.

Both models train on fully dense synthetic inputs (`event_density=1.0`). The sweep then evaluates on both dense and masked versions. There is no validation phase, and the scripts do not report training loss or save per-epoch checkpoints.

## Run sequence and generated files

From this folder, activate the environment where the dependencies were installed and run:

```bash
python experiments/train.py
python experiments/sparsity_sweep.py
python experiments/evaluate.py
```

The first command trains the models. It creates `results/latest/` if needed and writes:

- `ann.pt` and `snn.pt`: model parameter weights (state dictionaries).
- `metadata.json`: the configuration used and parameter counts. With the default architecture, the counts are 205,315 ANN parameters and 4,611 SNN parameters.

Training prints this metadata as JSON after training finishes. It may be quiet while the epochs are running.

The second command requires both weight files. It evaluates the models on 2,000 generated samples at each density and writes:

- `sparsity_results.csv`: one row per event density, convenient for spreadsheets and the dashboard.
- `sparsity_results.json`: the same rows in JSON format.

It also prints the table in the terminal. The third command reads `sparsity_results.csv` and prints it again. It does not retrain or recalculate the metrics. Add `--run some/path` to the sweep and evaluation commands to use a results folder other than `results/latest`.

To open the charts after the sweep:

```bash
streamlit run visualization/dashboard.py
```

The dashboard reads `results/latest/sparsity_results.csv` and shows compute-reduction, accuracy, and efficiency-ratio charts. It currently uses that fixed results path.

## What the result columns mean

Each row describes one test density. Accuracy and F1 are between 0 and 1; higher means better classification on this generated test set.

| Column | Meaning |
| --- | --- |
| `event_density` | Probability that each input position is retained by the synthetic mask. Lower means a sparser input. |
| `ann_accuracy`, `snn_accuracy` | Fraction of test samples classified correctly by the ANN or SNN. |
| `ann_f1`, `snn_f1` | Macro F1: F1 is calculated for each class and then averaged equally. This helps reveal when performance differs between classes. |
| `ann_macs_proxy` | Estimated ANN multiply-accumulate count for the full test set. It stays constant across densities because the ANN computation is treated as dense. |
| `snn_synops_proxy` | Code-defined estimate based on positive input values and output spikes. It is a rough simulation proxy, not a hardware instruction count. |
| `compute_reduction_proxy` | `1 - snn_synops_proxy / ann_macs_proxy`. Positive means the SNN proxy is lower than the ANN estimate; zero means equal; negative means the SNN proxy is higher. |
| `efficiency_ratio_proxy` | `ann_macs_proxy / snn_synops_proxy`. Above 1 means the ANN estimate is larger; below 1 means the SNN estimate is larger. |
| `snn_latency_ms` | Sum of measured SNN forward-pass time across test batches for that density. It depends on machine, software, and batch size. No ANN latency is recorded, so this is not a speed comparison. |
| `input_events` | Number of positive input values (`x > 0`) counted across the test set. It is not simply the number of nonzero values because negative values are excluded. |
| `output_spikes` | Total SNN output spikes across all test samples, time steps, and output classes. |

The SNN proxy is calculated from the number of positive input values times the hidden width, plus the number of output spikes times the hidden width. The ANN estimate counts the dense input-to-hidden and hidden-to-output matrix operations. These are deliberately simplified estimates and should not be presented as physical energy, actual hardware operations, or a guaranteed runtime advantage. In particular, the current PyTorch model still executes dense linear layers at every time step, even when many input values are zero.

## Reading the currently present example results

The `results/latest/sparsity_results.csv` currently in this folder reports this pattern (your next run can replace it):

| Density | ANN accuracy | SNN accuracy | Proxy reduction | What it suggests |
| ---: | ---: | ---: | ---: | --- |
| 1.00 | 1.000 | 1.000 | 48.0% | Both models classify the dense synthetic task well; the SNN proxy is lower in this run. |
| 0.50 | 1.000 | 0.772 | 74.2% | The SNN loses some accuracy as inputs are removed, while its proxy count also falls. |
| 0.25 | 1.000 | 0.350 | 87.1% | SNN accuracy is close to the 1-in-3 random baseline for three balanced classes. |
| 0.10 | 0.996 | 0.346 | 94.9% | The SNN has zero output spikes in this saved result; its predictions have effectively collapsed. |
| 0.05 | 0.952 | 0.337 | 97.4% | The SNN remains silent in this result and is near chance, despite the large proxy reduction. |
| 0.01 | 0.596 | 0.329 | 99.5% | Both models struggle with extremely sparse inputs; the SNN is silent and near chance. |

This illustrates the main trade-off the POC is intended to examine: a smaller operation proxy is useful only if task performance remains acceptable. Here the very large reductions at low density coincide with zero SNN output spikes and near-chance accuracy, so they do **not** demonstrate useful efficient inference at those densities. The ANN also degrades at the sparsest settings, though less sharply in this snapshot.

Latency does not fall in proportion to the proxy count in this result. That is expected from the current implementation: it measures ordinary PyTorch execution of dense layers and does not skip matrix operations for zero inputs. Timing also varies across machines and runs.

## Current scope and limitations

- The input data is synthetic and deliberately simple; results do not establish performance on a real application or dataset.
- Models are trained only with fully dense inputs, then tested at lower densities. Sparse-aware or mixed-density training is not implemented.
- The current event adapter and delta encoder are not connected to the training/evaluation pipeline.
- The hardware backend is only an interface. No neuromorphic device or energy meter is used.
- Compute reduction and efficiency ratio are operation-count proxies. They do not account for all costs, memory traffic, hardware sparsity support, or actual energy.
- The SNN sweep runs an SNN pass to collect benchmark metrics and another pass to produce predictions. The recorded latency measures only the benchmark pass.

The POC is therefore a starting benchmark scaffold: it can compare accuracy and simple operation proxies under controlled sparsity, while a practical efficiency claim would need stronger sparse-aware implementations and measurements on target hardware.
