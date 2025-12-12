# Weatherduck GNN Scaffold 🌦️🦆

This package contains a minimal, [fiddle](https://github.com/google/fiddle)-wired encode–process–decode graph neural network scaffold for weather-style data using PyTorch Geometric and Lightning.


## Why was this made?

Weatherduck was built to be a lightweight, hydra-free scaffold that mirrors [neural-lam](https://github.com/mllam/neural-lam) and [anemoi](https://github.com/ecmwf/anemoi-core)'s encode–process–decode GNN flow in pure Python/Fiddle. It’s designed to:

- Prototype message-passing architectures on weather-style data without wiring up the full neural-lam/anemoi stack.
- Serve as an inspiration for how one could structure GNN-based weather model achitectures and training in PyTorch Lightning + PyG + Fiddle.
  - See [example notebook](notebooks/fiddle.ipynb) using fiddle to visualize a weatherduck experiment
- Keep model architecture components small and override-friendly (for example with drop-in custom MessagePassing classes).
- Exercise end-to-end Lightning + PyG training with dummy graphs so you can iterate on model code and configs before real data/graphs are ready.
  - support for loading data from anemoi and neural-lam datasets is planned but not yet implemented.
- Clarify feature bookkeeping (n_*_features + trainable features) and graph expectations in one place.

## What’s inside
- `src/weatherduck/weatherduck.py`: Core implementation
  - `EncodeProcessDecodeModel`: encode → processor → decode GNN working on `HeteroData` batches.
  - `AutoRegressiveForecaster`: wraps a step-wise `EncodeProcessDecodeModel` to roll out multi-step forecasts over time.
  - `SingleNodesetEncoder`, `Processor`, `SingleNodesetDecoder`: slim mapper blocks.
  - `TrainableFeatures` + `TrainableFeatureManager`: learnable per-graph/node trainable features, shared across batches.
  - `WeatherDuckDataModule`/`DummyWeatherDataset`: placeholder static graphs for single-step training.
  - `TimeseriesWeatherDataModule`/`TimeseriesDummyWeatherDataset`: dummy timeseries graphs with node-first, time-last feature tensors for auto-regressive rollouts.
  - `experiment_factory`: Fiddle config returning an `Experiment` (single-step).
  - `autoregressive_experiment_factory`: Fiddle config for the autoregressive forecaster + timeseries data.
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
The model consumes PyG `HeteroData` with node types `{'data', 'hidden'}` and edges:
- `('data','to','hidden')`: encoder edges
- `('hidden','to','hidden')`: processor edges
- `('hidden','to','data')`: decoder edges
Edge attributes are optional; trainable features are repeated per graph when batches are stacked.

## Running tests
```bash
uv run python -m pytest weatherduck/tests/test_weatherduck.py
```
