import torch
from torch_geometric.nn import SAGEConv

from weatherduck.weatherduck import (
    EncodeProcessDecodeModel,
    SingleNodesetDecoder,
    SingleNodesetEncoder,
    WeatherDuckDataModule,
    Processor,
    build_dummy_weather_graph,
    make_mlp,
)


def test_single_batch_forward():
    n_input_data_features = 8
    n_output_data_features = 8
    n_hidden_data_features = 4
    n_input_trainable_features = 2
    n_trainable_hidden_features = 3
    hidden_dim = 64

    dm = WeatherDuckDataModule(
        num_samples=1,
        num_data_nodes=16,
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        batch_size=2,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    graph = build_dummy_weather_graph(
        num_data_nodes=dm.num_data_nodes,
        num_hidden_nodes=dm.num_data_nodes // 2,
        edge_attr_dim=2,
        n_data_node_features=0,
        n_hidden_node_features=n_hidden_data_features,
    )

    encoder = SingleNodesetEncoder(
        embedder_src=make_mlp(n_input_data_features + n_input_trainable_features, hidden_dim, hidden_dim),
        embedder_dst=make_mlp(n_hidden_data_features + n_trainable_hidden_features, hidden_dim, hidden_dim),
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        post_linear=torch.nn.Linear(hidden_dim, hidden_dim),
    )
    processor = Processor(
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        hidden_dim=hidden_dim,
    )
    decoder = SingleNodesetDecoder(
        embedder_src=make_mlp(
            hidden_dim + n_hidden_data_features + n_trainable_hidden_features, hidden_dim, hidden_dim
        ),
        embedder_dst=make_mlp(n_input_data_features + n_input_trainable_features, hidden_dim, hidden_dim),
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        out_linear=torch.nn.Linear(hidden_dim, n_output_data_features),
    )

    model = EncodeProcessDecodeModel(
        graph=graph,
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_output_data_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=n_input_trainable_features,
        n_trainable_hidden_features=n_trainable_hidden_features,
    )

    model.eval()
    with torch.no_grad():
        preds = model(batch)

    assert preds.shape == batch["data"].y.shape
