import torch

from weatherduck import (
    WeatherBackwardMapper,
    WeatherDuckDataModule,
    WeatherEncProcDec,
    WeatherForwardMapper,
    WeatherProcessor,
    build_dummy_weather_graph,
    make_mlp,
)
from torch_geometric.nn import SAGEConv


def test_single_batch_forward():
    in_channels_data = 8
    out_channels = 8
    trainable_data_dim = 2
    trainable_hidden_dim = 3
    hidden_dim = 64

    # DataModule and a single batch
    dm = WeatherDuckDataModule(
        num_samples=1,
        num_data_nodes=16,
        in_channels=in_channels_data,
        out_channels=out_channels,
        batch_size=1,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    # Graph template for model construction (features defined externally)
    graph = build_dummy_weather_graph(
        num_data_nodes=dm.num_data_nodes,
        num_hidden_nodes=dm.num_data_nodes // 2,
        edge_attr_dim=2,
        data_attr_dim=0,
        hidden_attr_dim=0,
    )

    encoder = WeatherForwardMapper(
        embed_src=make_mlp(in_channels_data + trainable_data_dim, hidden_dim, hidden_dim),
        embed_dst=make_mlp(trainable_hidden_dim, hidden_dim, hidden_dim),
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        post_linear=torch.nn.Linear(hidden_dim, hidden_dim),
    )
    processor = WeatherProcessor(
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        hidden_dim=hidden_dim,
    )
    decoder = WeatherBackwardMapper(
        embed_src=make_mlp(hidden_dim + trainable_hidden_dim, hidden_dim, hidden_dim),
        embed_dst=make_mlp(in_channels_data + trainable_data_dim, hidden_dim, hidden_dim),
        message_op=SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        out_linear=torch.nn.Linear(hidden_dim, out_channels),
    )

    model = WeatherEncProcDec(
        graph=graph,
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        in_channels_data=in_channels_data,
        out_channels=out_channels,
        trainable_data_dim=trainable_data_dim,
        trainable_hidden_dim=trainable_hidden_dim,
    )

    model.eval()
    with torch.no_grad():
        preds = model(batch)

    assert preds.shape == batch["data"].y.shape