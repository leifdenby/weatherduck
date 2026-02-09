import os
from dataclasses import dataclass
from pathlib import Path

import fiddle as fdl
import fiddle.experimental.auto_config
import pytest
import torch

from weatherduck import (
    AutoRegressiveForecaster,
    DummyGraphProvider,
    DummyTimeseriesWeatherDataModule,
    MDPDataModule,
    WMGGraphProvider,
    build_encode_process_decode_model,
)


@dataclass
class AutoregressiveExperiment:
    """Container for autoregressive smoke test components.

    Parameters
    ----------
    model : AutoRegressiveForecaster
        Autoregressive model to evaluate.
    data : DummyTimeseriesWeatherDataModule
        Data module providing autoregressive batches.

    Returns
    -------
    None
    """

    model: AutoRegressiveForecaster
    data: DummyTimeseriesWeatherDataModule

    def run(self) -> None:
        """Run a smoke test over a single batch.

        Returns
        -------
        None
        """
        self.data.setup("fit")
        batch = next(iter(self.data.train_dataloader()))
        self.model.eval()
        with torch.no_grad():
            preds = self.model(batch)
        assert preds.shape == batch["data"].y.shape


@fiddle.experimental.auto_config.auto_config
def experiment_factory() -> AutoregressiveExperiment:
    """Build a fiddle-configurable autoregressive experiment.

    Returns
    -------
    AutoregressiveExperiment
        Test experiment with dummy graph builder defaults.
    """
    ar_steps = 2
    n_state_features = 4
    n_output_features = 4
    n_hidden_data_features = 2
    n_input_trainable_features = 1
    n_hidden_trainable_features = 2
    hidden_dim = 32

    dm = DummyTimeseriesWeatherDataModule(
        graph_provider=DummyGraphProvider(),
        num_samples=4,
        num_data_nodes=8,
        n_state_features=n_state_features,
        n_forcing_features=2,
        n_static_features=1,
        ar_steps=ar_steps,
        n_hidden_data_features=n_hidden_data_features,
        batch_size=2,
        n_unique_graphs=2,
    )

    step_model = build_encode_process_decode_model(
        n_input_data_features=n_state_features + 2 + 1,  # state + forcing + static
        n_output_data_features=n_output_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=n_input_trainable_features,
        n_hidden_trainable_features=n_hidden_trainable_features,
        hidden_dim=hidden_dim,
    )
    ar_model = AutoRegressiveForecaster(step_predictor=step_model)

    return AutoregressiveExperiment(model=ar_model, data=dm)


def test_autoregressive_forecaster_runs_dummy_graph_provider():
    """Run the autoregressive forecaster using the dummy graph provider.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    config = experiment_factory()
    experiment = fdl.build(config)
    experiment.run()


def test_autoregressive_forecaster_runs_wmg_graph_provider():
    """Run the autoregressive forecaster using the WMG graph provider.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pytest.importorskip("weather_model_graphs")
    config = experiment_factory()
    config.data.graph_provider = WMGGraphProvider(
        kind="keisler", mesh_node_distance=1.0
    )
    experiment = fdl.build(config)
    experiment.run()


def test_autoregressive_forecaster_runs_neural_lam_datamodule():
    """Run the autoregressive forecaster using the neural-lam MDP datamodule.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pytest.importorskip("neural_lam")
    from weatherduck.graphs import WMGGraphProvider

    default_config = (
        Path(__file__).resolve().parent
        / "data"
        / "neural_lam"
        / "mdp"
        / "danra_100m_winds"
        / "danra.datastore.yaml"
    )
    config_path = Path(os.environ.get("NEURAL_LAM_MDP_CONFIG", default_config))
    if not config_path.exists():
        pytest.skip(f"Neural-lam config not found at {config_path}.")

    config = experiment_factory()
    config.data = MDPDataModule(
        config_path=str(config_path),
        graph_provider=WMGGraphProvider(mesh_node_distance=10.0e3),
        ar_steps_train=2,
        ar_steps_eval=2,
        batch_size=2,
        num_workers=0,
    )
    experiment = fdl.build(config)
    experiment.run()
