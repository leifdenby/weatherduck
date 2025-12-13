# AGENTS GUIDE

Notes for future agents working in this repo.

- **Purpose**: Encode–Process–Decode GNN scaffold with an optional auto-regressive forecaster, built on PyTorch Geometric, PyTorch Lightning, and Fiddle.
- **Key module**: `src/weatherduck/weatherduck.py` houses:
  - Model components: `SingleNodesetEncoder`, `Processor`, `SingleNodesetDecoder`, `EncodeProcessDecodeModel`.
  - Trainable feature utilities: `TrainableFeatures`, `TrainableFeatureManager`.
  - Auto-regressive wrapper: `AutoRegressiveForecaster`.
- Dummy datasets/datamodules (single-step and timeseries), `WeatherDuckModule`, and Fiddle factories (`build_encode_process_decode_model`, `experiment_factory`, `autoregressive_experiment_factory`).
  - CLI entrypoint in `src/weatherduck/main.py` (run with `uv run weatherduck`).
- **Dependencies/tooling**: Managed via `uv` (`pyproject.toml` + `uv.lock`). Use `uv run ...` to respect the locked environment.
- **Testing**: Run the suite with `uv run pytest`.
- **Fiddle caveat**: `auto_config` forbids control-flow (loops/comprehensions) inside decorated factory functions; keep them simple.
- **Docs/examples**: `README.md` covers usage and graph expectations; see `tests/` for working model and autoregressive flows.
