# Weatherduck GNN Scaffold 🌦️🦆

This package contains a minimal, [fiddle](https://github.com/google/fiddle)-wired encode–process–decode graph neural network scaffold for weather-style data using PyTorch Geometric and Lightning.


## Why was this made?

Weatherduck was built to be a lightweight, hydra-free scaffold that mirrors [neural-lam](https://github.com/mllam/neural-lam) and [anemoi](https://github.com/ecmwf/anemoi-core)'s encode–process–decode GNN flow in pure Python/Fiddle. It’s designed to:

- Prototype message-passing architectures on weather-style data without wiring up the full neural-lam/anemoi stack.
- Serve as an inspiration for how one could structure GNN-based weather model achitectures and training in PyTorch Lightning + PyG + Fiddle.
  - See [example notebook](notebooks/fiddle.ipynb) using fiddle to visualize a weatherduck experiment
- Keep model architecture components small and override-friendly (for example with drop-in custom MessagePassing classes).
- Exercise end-to-end Lightning + PyG training with dummy graphs so you can iterate on model code and configs before real data/graphs are ready.
  - includes a NativeGridDataset-backed Anemoi datamodule for real datasets.
- Clarify feature bookkeeping (n_*_features + trainable features) and graph expectations in one place.

## What’s inside
- `src/weatherduck/step_predictor.py`: single-step components (`EncodeProcessDecodeModel`, `SingleNodesetEncoder`/`Processor`/`SingleNodesetDecoder`, trainable feature utilities).
- `src/weatherduck/lightning.py`: Lightning wrapper (`WeatherDuckModule`) around any model.
- `src/weatherduck/ar_forecaster.py`: `AutoRegressiveForecaster` that rolls out multi-step predictions with a provided step predictor.
- `src/weatherduck/data/dummy.py`: dummy datasets/datamodules for single-step and timeseries graphs plus `build_dummy_weather_graph`.
- `src/weatherduck/configs.py`: Fiddle factories (`build_encode_process_decode_model`, `experiment_factory`, `autoregressive_experiment_factory`) and the `Experiment` dataclass.
- `src/weatherduck/__init__.py`: Public exports.
- `tests/test_weatherduck.py`: Smoke tests for single-step training.
- `tests/test_autoregressive.py`: Smoke tests for autoregressive forecasting.
- `main.py` (invoked by `uv run weatherduck`): builds the Fiddle experiment and runs a short training loop.

## Quick start
```bash
uv run weatherduck  # runs experiment_factory → Experiment.run()
```
This uses dummy graphs/data and should execute end-to-end on CPU or MPS.

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

## Running tests
```bash
uv run pytest
```

## Anemoi dataset config example (Fiddle)
Below is a minimal, Fiddle-friendly dataset config focused only on the data
portion of Anemoi's limited-area setup. It mirrors the `dataloader.dataset`
shape from `anemoi-core/training/src/anemoi/training/config/lam.yaml` and the
train/val/test split fields from `anemoi-core/training/src/anemoi/training/config/dataloader/native_grid.yaml`.

```python
import fiddle as fdl

from weatherduck.data.anemoi import AnemoiNativeGridDataModule


def build_anemoi_dataset_splits():
    base_dataset = {
        "cutout": [
            {"dataset": "/path/to/main.zarr", "thinning": 4},
            {"dataset": "/path/to/forcing.zarr"},
        ],
        "adjust": "all",
        "min_distance_km": 0,
    }
    training = {
        "dataset": base_dataset,
        "start": 2020,
        "end": 2020,
        "frequency": "6h",
        "drop": [],
    }
    validation = {
        "dataset": base_dataset,
        "start": 2021,
        "end": 2021,
        "frequency": "6h",
        "drop": [],
    }
    test = {
        "dataset": base_dataset,
        "start": 2022,
        "end": None,
        "frequency": "6h",
        "drop": [],
    }
    return training, validation, test


training, validation, test = build_anemoi_dataset_splits()

datamodule_cfg = fdl.Config(
    AnemoiNativeGridDataModule,
    graph_data=graph,
    training=training,
    validation=validation,
    test=test,
    data_frequency="6h",
    data_timestep="6h",
    multistep_input=1,
    rollout=1,
    validation_rollout=1,
)
```
