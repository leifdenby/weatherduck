from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader as GeoDataLoader


def build_dummy_weather_graph(
    num_data_nodes: int = 64,
    num_hidden_nodes: int = 32,
    edge_attr_dim: int = 2,
    n_data_node_features: int = 0,
    n_hidden_node_features: int = 0,
) -> HeteroData:
    """
    Build a minimal heterogeneous graph with the expected topology:
    data -> hidden, hidden -> hidden, hidden -> data.
    """
    graph = HeteroData()
    graph["data"].x = torch.randn(num_data_nodes, n_data_node_features)
    graph["hidden"].x = torch.randn(num_hidden_nodes, n_hidden_node_features)

    def dense_edges(n_src: int, n_dst: int, fanout: int) -> torch.Tensor:
        src = torch.arange(n_src).repeat_interleave(fanout)
        dst_choices = torch.randint(0, n_dst, (n_src * fanout,))
        return torch.stack([src, dst_choices], dim=0)

    graph["data", "to", "hidden"].edge_index = dense_edges(num_data_nodes, num_hidden_nodes, fanout=4)
    graph["hidden", "to", "hidden"].edge_index = dense_edges(num_hidden_nodes, num_hidden_nodes, fanout=6)
    graph["hidden", "to", "data"].edge_index = dense_edges(num_hidden_nodes, num_data_nodes, fanout=4)

    for key in [
        ("data", "to", "hidden"),
        ("hidden", "to", "hidden"),
        ("hidden", "to", "data"),
    ]:
        num_edges = graph[key].edge_index.shape[1]
        graph[key].edge_attr = torch.randn(num_edges, edge_attr_dim)

    return graph


class DummyWeatherDataset(Dataset):
    """
    Dummy dataset producing random HeteroData samples for quick execution.
    """

    def __init__(
        self,
        num_samples: int,
        num_data_nodes: int | dict[int, int],
        n_input_data_features: int,
        n_output_data_features: int,
        n_hidden_data_features: int,
        n_unique_graphs: int = 1,
    ):
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_input_data_features = n_input_data_features
        self.n_output_data_features = n_output_data_features
        self.n_hidden_data_features = n_hidden_data_features
        self.n_unique_graphs = n_unique_graphs
        self.graphs: list[HeteroData] = []
        for gid in range(n_unique_graphs):
            if isinstance(self.num_data_nodes, dict):
                num_nodes = self.num_data_nodes[gid]
            else:
                num_nodes = self.num_data_nodes
            g = build_dummy_weather_graph(
                num_data_nodes=num_nodes,
                num_hidden_nodes=max(1, num_nodes // 2),
                edge_attr_dim=2,
                n_data_node_features=0,
                n_hidden_node_features=self.n_hidden_data_features,
            )
            g.graph_id = torch.tensor([gid], dtype=torch.long)
            self.graphs.append(g)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> HeteroData:
        graph = self.graphs[idx % self.n_unique_graphs].clone()
        if isinstance(self.num_data_nodes, dict):
            gid = int(graph.graph_id.item())
            num_data_nodes = self.num_data_nodes.get(gid)
            assert num_data_nodes is not None, f"num_data_nodes missing entry for graph id {gid}"
        else:
            num_data_nodes = self.num_data_nodes

        graph["data"].x = torch.randn(num_data_nodes, self.n_input_data_features)
        if self.n_hidden_data_features > 0:
            graph["hidden"].x = torch.randn(graph["hidden"].num_nodes, self.n_hidden_data_features)
        else:
            graph["hidden"].x = torch.zeros(graph["hidden"].num_nodes, 0)
        graph["data"].y = torch.randn(num_data_nodes, self.n_output_data_features)
        return graph

    def collate_fn(self, graphs: list[HeteroData]) -> Batch:
        return Batch.from_data_list(graphs)


class TimeseriesDummyWeatherDataset(Dataset):
    """
    Dummy dataset producing HeteroData with timeseries splits for the
    AutoRegressiveForecaster.
    """

    def __init__(
        self,
        num_samples: int,
        num_data_nodes: int | dict[int, int],
        n_state_features: int,
        n_forcing_features: int,
        n_static_features: int,
        ar_steps: int,
        n_hidden_data_features: int,
        n_unique_graphs: int = 1,
    ):
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_state_features = n_state_features
        self.n_forcing_features = n_forcing_features
        self.n_static_features = n_static_features
        self.ar_steps = ar_steps
        self.n_hidden_data_features = n_hidden_data_features
        self.n_unique_graphs = n_unique_graphs

        self.graphs: list[HeteroData] = []
        for gid in range(n_unique_graphs):
            num_nodes = num_data_nodes[gid] if isinstance(num_data_nodes, dict) else num_data_nodes
            g = build_dummy_weather_graph(
                num_data_nodes=num_nodes,
                num_hidden_nodes=max(1, num_nodes // 2),
                edge_attr_dim=2,
                n_data_node_features=0,
                n_hidden_node_features=self.n_hidden_data_features,
            )
            g.graph_id = torch.tensor([gid], dtype=torch.long)
            self.graphs.append(g)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> HeteroData:
        graph = self.graphs[idx % self.n_unique_graphs].clone()
        gid = int(graph.graph_id.item())
        num_nodes = self.num_data_nodes[gid] if isinstance(self.num_data_nodes, dict) else self.num_data_nodes

        graph["data"].x_init_states = torch.randn(num_nodes, self.n_state_features, 2)
        graph["data"].x_target_states = torch.randn(num_nodes, self.n_state_features, self.ar_steps)
        graph["data"].x_forcing = torch.randn(num_nodes, self.n_forcing_features, self.ar_steps)
        graph["data"].x_static = torch.randn(num_nodes, self.n_static_features)
        graph["data"].x = graph["data"].x_init_states[:, :, -1]
        graph["data"].y = graph["data"].x_target_states  # [N, d_state, T]

        if self.n_hidden_data_features > 0:
            graph["hidden"].x = torch.randn(graph["hidden"].num_nodes, self.n_hidden_data_features)
        else:
            graph["hidden"].x = torch.zeros(graph["hidden"].num_nodes, 0)
        return graph

    def collate_fn(self, graphs: list[HeteroData]) -> Batch:
        return Batch.from_data_list(graphs)


class WeatherDuckDataModule(pl.LightningDataModule):
    """
    LightningDataModule providing dummy weather graphs via PyG DataLoader.
    """

    def __init__(
        self,
        num_samples: int = 128,
        num_data_nodes: int | dict[int, int] = 64,
        n_input_data_features: int = 8,
        n_output_data_features: int = 8,
        n_hidden_data_features: int = 0,
        batch_size: int = 4,
        n_unique_graphs: int = 1,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_input_data_features = n_input_data_features
        self.n_output_data_features = n_output_data_features
        self.n_hidden_data_features = n_hidden_data_features
        self.batch_size = batch_size
        self.n_unique_graphs = n_unique_graphs

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = DummyWeatherDataset(
            num_samples=self.num_samples,
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            n_hidden_data_features=self.n_hidden_data_features,
            n_unique_graphs=self.n_unique_graphs,
        )
        self.val_ds = DummyWeatherDataset(
            num_samples=max(8, self.num_samples // 10),
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            n_hidden_data_features=self.n_hidden_data_features,
            n_unique_graphs=self.n_unique_graphs,
        )
        self.test_ds = DummyWeatherDataset(
            num_samples=max(8, self.num_samples // 10),
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            n_hidden_data_features=self.n_hidden_data_features,
            n_unique_graphs=self.n_unique_graphs,
        )

    def train_dataloader(self) -> GeoDataLoader:
        return GeoDataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_ds.collate_fn)

    def val_dataloader(self) -> GeoDataLoader:
        return GeoDataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self.val_ds.collate_fn)

    def test_dataloader(self) -> GeoDataLoader:
        return GeoDataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.test_ds.collate_fn)


class TimeseriesWeatherDataModule(pl.LightningDataModule):
    """
    DataModule for timeseries dummy data compatible with AutoRegressiveForecaster.
    """

    def __init__(
        self,
        num_samples: int = 128,
        num_data_nodes: int | dict[int, int] = 64,
        n_state_features: int = 4,
        n_forcing_features: int = 2,
        n_static_features: int = 1,
        ar_steps: int = 3,
        n_hidden_data_features: int = 0,
        batch_size: int = 4,
        n_unique_graphs: int = 1,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_state_features = n_state_features
        self.n_forcing_features = n_forcing_features
        self.n_static_features = n_static_features
        self.ar_steps = ar_steps
        self.n_hidden_data_features = n_hidden_data_features
        self.batch_size = batch_size
        self.n_unique_graphs = n_unique_graphs

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = TimeseriesDummyWeatherDataset(
            num_samples=self.num_samples,
            num_data_nodes=self.num_data_nodes,
            n_state_features=self.n_state_features,
            n_forcing_features=self.n_forcing_features,
            n_static_features=self.n_static_features,
            ar_steps=self.ar_steps,
            n_hidden_data_features=self.n_hidden_data_features,
            n_unique_graphs=self.n_unique_graphs,
        )
        self.val_ds = TimeseriesDummyWeatherDataset(
            num_samples=max(8, self.num_samples // 10),
            num_data_nodes=self.num_data_nodes,
            n_state_features=self.n_state_features,
            n_forcing_features=self.n_forcing_features,
            n_static_features=self.n_static_features,
            ar_steps=self.ar_steps,
            n_hidden_data_features=self.n_hidden_data_features,
            n_unique_graphs=self.n_unique_graphs,
        )
        self.test_ds = TimeseriesDummyWeatherDataset(
            num_samples=max(8, self.num_samples // 10),
            num_data_nodes=self.num_data_nodes,
            n_state_features=self.n_state_features,
            n_forcing_features=self.n_forcing_features,
            n_static_features=self.n_static_features,
            ar_steps=self.ar_steps,
            n_hidden_data_features=self.n_hidden_data_features,
            n_unique_graphs=self.n_unique_graphs,
        )

    def train_dataloader(self) -> GeoDataLoader:
        return GeoDataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_ds.collate_fn)

    def val_dataloader(self) -> GeoDataLoader:
        return GeoDataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self.val_ds.collate_fn)

    def test_dataloader(self) -> GeoDataLoader:
        return GeoDataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.test_ds.collate_fn)
