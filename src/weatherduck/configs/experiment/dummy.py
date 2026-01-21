import fiddle
import fiddle.experimental
import fiddle.experimental.auto_config
import pytorch_lightning as pl

from ...ar_forecaster import AutoRegressiveForecaster
from ...data.dummy import TimeseriesWeatherDataModule, WeatherDuckDataModule
from ...lightning import WeatherDuckModule
from ..model.base import build_encode_process_decode_model
from .base import Experiment

__all__ = ["Experiment", "experiment_factory", "autoregressive_experiment_factory"]


@fiddle.experimental.auto_config.auto_config
def experiment_factory() -> Experiment:
    """
    Build a experiment object for a dummy single-step weather prediction task.
    This is decorated as a Fiddle auto_config function, so that one can create
    a buildable config for the experiment where experiment setup can
    overridden.

    Returns:
        Experiment containing the model, data module, and trainer config.
    """
    n_input_data_features = 8
    n_output_data_features = 8
    hidden_dim = 128
    n_hidden_data_features = 4
    n_input_trainable_features = 2
    n_hidden_trainable_features = 3
    core_model = build_encode_process_decode_model(
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=n_input_trainable_features,
        n_hidden_trainable_features=n_hidden_trainable_features,
        hidden_dim=hidden_dim,
    )

    lit_module = WeatherDuckModule(
        model=core_model,
        lr=1e-3,
    )

    data = WeatherDuckDataModule(
        num_samples=256,
        num_data_nodes=64,
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        batch_size=4,
    )

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1,
    )

    return Experiment(
        pl_module=lit_module,
        data=data,
        trainer=trainer,
    )


@fiddle.experimental.auto_config.auto_config
def autoregressive_experiment_factory() -> Experiment:
    """Build a Fiddle config graph for the autoregressive dummy experiment.

    Returns:
        Experiment containing the autoregressive model, data, and trainer config.
    """
    ar_steps = 3
    n_state_features = 6
    n_output_data_features = 6
    n_hidden_data_features = 3
    n_input_trainable_features = 2
    n_hidden_trainable_features = 2
    n_forcing_features = 2
    n_static_features = 1
    hidden_dim = 128

    step_model = build_encode_process_decode_model(
        n_input_data_features=n_state_features
        + n_forcing_features
        + n_static_features,  # state + forcing + static
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=n_input_trainable_features,
        n_hidden_trainable_features=n_hidden_trainable_features,
        hidden_dim=hidden_dim,
    )

    ar_model = AutoRegressiveForecaster(
        step_predictor=step_model,
    )

    lit_module = WeatherDuckModule(
        model=ar_model,
        lr=1e-3,
    )

    data = TimeseriesWeatherDataModule(
        num_samples=256,
        num_data_nodes=64,
        n_state_features=n_state_features,
        n_forcing_features=n_forcing_features,
        n_static_features=n_static_features,
        ar_steps=ar_steps,
        n_hidden_data_features=n_hidden_data_features,
        batch_size=4,
        n_unique_graphs=2,
    )

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1,
    )

    return Experiment(
        pl_module=lit_module,
        data=data,
        trainer=trainer,
    )
