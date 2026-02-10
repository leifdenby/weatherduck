# Weatherduck GNN Scaffold 🌦️🦆

This package contains a minimal, [fiddle](https://github.com/google/fiddle)-wired encode–process–decode graph neural network scaffold for weather-style data using PyTorch Geometric and Lightning.


## Why was this made?

Weatherduck was built to be a lightweight, hydra-free scaffold that mirrors [neural-lam](https://github.com/mllam/neural-lam) and [anemoi](https://github.com/ecmwf/anemoi-core)'s encode–process–decode GNN flow in pure Python/Fiddle. It’s designed to:

- Prototype message-passing architectures on weather-style data without wiring up the full neural-lam/anemoi stack.
- Serve as an inspiration for how one could structure GNN-based weather model achitectures and training in PyTorch Lightning + PyG + Fiddle.
  - See [example notebook](notebooks/fiddle.ipynb) using fiddle to visualize a weatherduck experiment
- Keep model architecture components small and override-friendly (for example with drop-in custom MessagePassing classes).
- Exercise end-to-end Lightning + PyG training with dummy graphs so you can iterate on model code and configs before real data/graphs are ready.
  - neural-lam MDP datasets are supported via `MDPDataModule`; anemoi dataset support is still planned.
- Clarify feature bookkeeping (n_*_features + trainable features) and graph expectations in one place.

## What’s inside
- `src/weatherduck/step_predictor.py`: single-step components (`EncodeProcessDecodeModel`, `SingleNodesetEncoder`/`Processor`/`SingleNodesetDecoder`, trainable feature utilities).
- `src/weatherduck/lightning.py`: Lightning wrapper (`WeatherDuckModule`) around any model.
- `src/weatherduck/ar_forecaster.py`: `AutoRegressiveForecaster` that rolls out multi-step predictions with a provided step predictor.
- `src/weatherduck/graphs/`: graph provider interfaces and implementations (`GraphProvider`, `DummyGraphProvider`, `WMGGraphProvider`).
- `src/weatherduck/data/base.py`: `BaseWeatherDataModule` shared datamodule base class that standardizes split handling and GeoDataLoader wiring for datasets that yield `HeteroData`.
- `src/weatherduck/data/dummy.py`: dummy dataset implementations and `DummyWeatherDataModule`/`DummyTimeseriesWeatherDataModule` (built on the base class).
- `src/weatherduck/data/neural_lam.py`: neural-lam-backed `MDPDataModule` and dataset adapter.
- `src/weatherduck/configs.py`: Fiddle factories (`build_encode_process_decode_model`, `experiment_factory`, `autoregressive_experiment_factory`) and the `Experiment` dataclass.
- `src/weatherduck/__init__.py`: Public exports.
- `tests/test_weatherduck.py`: Smoke tests for single-step training.
- `tests/test_autoregressive.py`: Smoke tests for autoregressive forecasting.
- `main.py` (invoked by `uv run weatherduck`): builds the Fiddle experiment and runs a short training loop.

## Get started (local)
Clone the repo and install dependencies with `uv` (recommended) or `pip -e .`:

```bash
git clone <repo-url>
cd weatherduck

# Option 1: uv
uv sync --dev --all-extras
uv run weatherduck  # runs experiment_factory → Experiment.run()

# Option 2: editable install with pip
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
weatherduck  # runs experiment_factory → Experiment.run()
```

Optional dependency groups are required for graph providers and datasets:
- `wmg`: enables `WMGGraphProvider` ([weather-model-graphs](https://github.com/mllam/weather-model-graphs)).
- `neural-lam`: enables `MDPDataModule` and neural-lam datasets ([neural-lam](https://github.com/mllam/neural-lam)).

With `uv`:
```bash
uv sync --dev --all-extras --group wmg
uv sync --dev --all-extras --group neural-lam
```

With `pip`:
```bash
pip install -e .[dev,wmg]
pip install -e .[dev,neural-lam]
```

By default [`experiment_factory`](src/weatherduck/configs.py) uses dummy graphs/data and should execute end-to-end on CPU or MPS.


## Key dimensions (n_*)
- `n_input_data_features`: dataset-provided data-node features.
- `n_hidden_data_features`: dataset-provided hidden-node features.
- `n_input_trainable_features`: learnable features appended to each data node.
- `n_hidden_trainable_features`: learnable features appended to each hidden node.
- `n_output_data_features`: decoder output channels on data nodes.

## Graph expectations
`EncodeProcessDecodeModel`:
- Node types: `{'data', 'hidden'}` with
  - `graph['data'].x`: `[N_data, n_input_data_features]`
  - `graph['hidden'].x`: `[N_hidden, n_hidden_data_features]`
- Edge types:
  - `('data','to','hidden')` with `edge_index` `[2, E_dh]` (optional `edge_attr`)
  - `('hidden','to','hidden')` with `edge_index` `[2, E_hh]` (optional `edge_attr`)
  - `('hidden','to','data')` with `edge_index` `[2, E_hd]` (optional `edge_attr`)
- Trainable features (if enabled) are added per graph and concatenated to the corresponding node features.

`AutoRegressiveForecaster` (wraps e.g. an `EncodeProcessDecodeModel` for one-step prediction)
- Node type: `{'data'}` features:
  - `x_init_states`: `[N, d_state, 2]` initial history (latest state in the last slot)
  - `x_forcing`: `[N, d_forcing, T]`
  - `x_static`: `[N, d_static]`
- Shares the same edge/node structure required by the underlying EncodeProcessDecodeModel (data/hidden node types and the three edge sets above). From `x_init_states`, `x_forcing` and `x_static` the model constructs `graph["data"].x` for each step to pass down to the provided `step_predictor` (e.g. an EncodeProcessDecodeModel).

`WeatherDuckModule` (`LightningModule`) (takes e.g. an `EncodeProcessDecodeModel` or `AutoRegressiveForecaster`)
- passes `graph` to the model's `forward` method to get predictions (`y_hat`)
- expects `graph['data'].y` to compute the loss; this tensor is not consumed by the step predictor itself.

Shapes follow the convention: first dim = nodes, last dim = time (for sequences), this is required because PyG data-loader batches graphs along the first dimension.

## Graph providers
Weatherduck constructs graphs through a `GraphProvider` interface. A `GraphProvider`
is a small callable that takes a `domain_id` and data-node coordinates, then returns
a `HeteroData` graph matching Weatherduck’s expected node/edge types and shapes.
This keeps graph construction independent from datasets and lets you swap in
different graph generators without changing data pipelines.

`GraphProvider` includes an optional in-memory cache. Providers combine their
configuration with the `domain_id` to compute a stable graph id for caching, so
repeat calls for the same domain can reuse the graph instead of recomputing it.

Implementations:
- `DummyGraphProvider`: produces a minimal hetero graph for quick iteration.
- `WMGGraphProvider`: uses `weather-model-graphs` to build a graph from spatial coordinates.

Example:
```python
from weatherduck import DummyGraphProvider, DummyWeatherDataModule

dm = DummyWeatherDataModule(
    graph_provider=DummyGraphProvider(),
    num_samples=64,
    num_data_nodes=64,
    n_input_data_features=8,
    n_output_data_features=8,
    n_hidden_data_features=4,
)
```

## Running tests
```bash
uv run pytest
```
