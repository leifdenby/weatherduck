from dataclasses import dataclass

import fiddle
import fiddle.experimental
import fiddle.experimental.auto_config
import pytorch_lightning as pl
from torch import nn
from torch_geometric.nn import SAGEConv

from .ar_forecaster import AutoRegressiveForecaster
from .data.dummy import TimeseriesWeatherDataModule, WeatherDuckDataModule
from .lightning import WeatherDuckModule
from .step_predictor import (
    EncodeProcessDecodeModel,
    Processor,
    SingleNodesetDecoder,
    SingleNodesetEncoder,
    TrainableFeatureManager,
    make_mlp,
)

__all__ = [
    "Experiment",
    "build_encode_process_decode_model",
    "experiment_factory",
    "autoregressive_experiment_factory",
]


@dataclass
class Experiment:
    model: pl.LightningModule
    data: pl.LightningDataModule
    trainer: pl.Trainer

    def run(self) -> None:
        self.trainer.fit(self.model, datamodule=self.data)
        self.trainer.test(self.model, datamodule=self.data)


@fiddle.experimental.auto_config.auto_config
def build_encode_process_decode_model(
    *,
    n_input_data_features: int,
    n_output_data_features: int,
    n_hidden_data_features: int,
    n_input_trainable_features: int,
    n_hidden_trainable_features: int,
    hidden_dim: int,
) -> EncodeProcessDecodeModel:
    """
    Factory to build an EncodeProcessDecodeModel with SAGEConv components.
    """
    encoder = SingleNodesetEncoder(
        embedder_src=make_mlp(
            n_input_data_features + n_input_trainable_features, hidden_dim, hidden_dim
        ),
        embedder_dst=make_mlp(
            n_hidden_data_features + n_hidden_trainable_features, hidden_dim, hidden_dim
        ),
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        post_linear=nn.Linear(hidden_dim, hidden_dim),
    )
    processor = Processor(
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        hidden_dim=hidden_dim,
    )
    decoder = SingleNodesetDecoder(
        embedder_src=make_mlp(
            hidden_dim + n_hidden_data_features + n_hidden_trainable_features,
            hidden_dim,
            hidden_dim,
        ),
        embedder_dst=make_mlp(
            n_input_data_features + n_input_trainable_features, hidden_dim, hidden_dim
        ),
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        out_linear=nn.Linear(hidden_dim, n_output_data_features),
    )

    trainable_manager = TrainableFeatureManager(
        n_input_trainable_features, n_hidden_trainable_features
    )

    return EncodeProcessDecodeModel(
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=n_input_trainable_features,
        n_hidden_trainable_features=n_hidden_trainable_features,
        trainable_manager=trainable_manager,
    )


@fiddle.experimental.auto_config.auto_config
def experiment_factory() -> Experiment:
    """
    Build a Fiddle config graph that mirrors the Hydra GNN setup
    (encode -> process -> decode) but runs with dummy graph/data.
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
        model=lit_module,
        data=data,
        trainer=trainer,
    )


@fiddle.experimental.auto_config.auto_config
def autoregressive_experiment_factory() -> Experiment:
    """
    Build a Fiddle config graph for the autoregressive forecaster with
    timeseries dummy data.
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
        model=lit_module,
        data=data,
        trainer=trainer,
    )
