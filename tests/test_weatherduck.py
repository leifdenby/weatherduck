import torch

from weatherduck import (
    WeatherDuckDataModule,
    build_encode_process_decode_model,
    build_dummy_weather_graph,
)


def test_single_batch_forward():
    """Run a single batch through the model and check output shape matches targets."""
    n_input_data_features = 8
    n_output_data_features = 8
    n_hidden_data_features = 4
    n_input_trainable_features = 2
    n_hidden_trainable_features = 3
    hidden_dim = 64

    dm = WeatherDuckDataModule(
        num_samples=1,
        num_data_nodes=16,
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        batch_size=2,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    model = build_encode_process_decode_model(
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=n_input_trainable_features,
        n_hidden_trainable_features=n_hidden_trainable_features,
        hidden_dim=hidden_dim,
    )

    model.eval()
    with torch.no_grad():
        preds = model(batch)

    assert preds.shape == batch["data"].y.shape


def test_trainable_params_match_unique_graphs():
    """Ensure per-graph trainable modules are created for each unique graph in the dataset."""
    n_input_data_features = 4
    n_output_data_features = 2
    n_hidden_data_features = 1
    n_input_trainable_features = 2
    n_hidden_trainable_features = 3
    hidden_dim = 16
    num_nodes_per_graph = {0: 6, 1: 8, 2: 10}
    n_unique_graphs = len(num_nodes_per_graph)

    dm = WeatherDuckDataModule(
        num_samples=6,
        num_data_nodes=num_nodes_per_graph,
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        batch_size=2,
        n_unique_graphs=n_unique_graphs,
    )
    dm.setup("fit")
    model = build_encode_process_decode_model(
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=n_input_trainable_features,
        n_hidden_trainable_features=n_hidden_trainable_features,
        hidden_dim=hidden_dim,
    )

    model.eval()
    with torch.no_grad():
        for batch in dm.train_dataloader():
            _ = model(batch)

    # Trainable feature modules should match number of unique graphs
    manager = model.trainable_manager
    assert len(manager.data_modules) == n_unique_graphs
    assert len(manager.hidden_modules) == n_unique_graphs
    for gid, module in manager.data_modules.items():
        expected = num_nodes_per_graph[int(gid)]
        assert module.trainable.shape[0] == expected
    for gid, module in manager.hidden_modules.items():
        expected = num_nodes_per_graph[int(gid)] // 2
        assert module.trainable.shape[0] == expected
