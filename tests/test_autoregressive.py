import torch
from torch_geometric.nn import SAGEConv

from weatherduck.weatherduck import (
    AutoRegressiveForecaster,
    TimeseriesWeatherDataModule,
    build_encode_process_decode_model,
)


def test_autoregressive_forecaster_runs():
    ar_steps = 2
    n_state_features = 4
    n_output_features = 4
    n_hidden_data_features = 2
    n_input_trainable_features = 1
    n_hidden_trainable_features = 2
    hidden_dim = 32

    dm = TimeseriesWeatherDataModule(
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
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    # step model: input = current state + forcing + static; we fold forcing/static dims into input_data_features here
    step_model = build_encode_process_decode_model(
        n_input_data_features=n_state_features + 2 + 1,  # state + forcing + static
        n_output_data_features=n_output_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=n_input_trainable_features,
        n_hidden_trainable_features=n_hidden_trainable_features,
        hidden_dim=hidden_dim,
    )

    ar_model = AutoRegressiveForecaster(
        step_predictor=step_model,
    )

    ar_model.eval()
    with torch.no_grad():
        preds = ar_model(batch)

    assert preds.shape == (batch["data"].num_nodes, n_output_features, ar_steps)
