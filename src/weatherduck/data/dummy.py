from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader as GeoDataLoader

from ..graphs import GraphBuilder


def _make_grid_coords(num_nodes: int) -> np.ndarray:
    """Create a square-ish grid of coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of coordinate pairs to generate.

    Returns
    -------
    np.ndarray
        Coordinates of shape [num_nodes, 2].
    """
    side = int(np.ceil(np.sqrt(num_nodes)))
    xv, yv = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)
    return coords[:num_nodes].astype(float)


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
        graph_builder: GraphBuilder,
        n_unique_graphs: int = 1,
    ):
        """Initialize the dummy dataset.

        Parameters
        ----------
        num_samples : int
            Number of samples in the dataset.
        num_data_nodes : int | dict[int, int]
            Number of data nodes per graph.
        n_input_data_features : int
            Input feature dimension.
        n_output_data_features : int
            Output feature dimension.
        n_hidden_data_features : int
            Hidden node feature dimension.
        graph_builder : GraphBuilder
            Graph builder used to create topology.
        n_unique_graphs : int, optional
            Number of unique graphs, by default 1.

        Returns
        -------
        None
        """
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_input_data_features = n_input_data_features
        self.n_output_data_features = n_output_data_features
        self.n_hidden_data_features = n_hidden_data_features
        self.graph_builder = graph_builder
        self.n_unique_graphs = n_unique_graphs
        self.graphs: list[HeteroData] = []
        for gid in range(n_unique_graphs):
            if isinstance(self.num_data_nodes, dict):
                num_nodes = self.num_data_nodes[gid]
            else:
                num_nodes = self.num_data_nodes
            coords = _make_grid_coords(num_nodes)
            if coords.shape[0] != num_nodes:
                raise ValueError(
                    f"Generated coords has {coords.shape[0]} nodes but dataset expects {num_nodes}."
                )
            g = self.graph_builder(coords)
            g.graph_id = torch.tensor([gid], dtype=torch.long)
            self.graphs.append(g)

    def __len__(self) -> int:
        """Return dataset length.

        Returns
        -------
        int
            Number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> HeteroData:
        """Return a single dummy graph sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        HeteroData
            Graph with populated data features/targets.
        """
        graph = self.graphs[idx % self.n_unique_graphs].clone()
        if isinstance(self.num_data_nodes, dict):
            gid = int(graph.graph_id.item())
            num_data_nodes = self.num_data_nodes.get(gid)
            assert (
                num_data_nodes is not None
            ), f"num_data_nodes missing entry for graph id {gid}"
        else:
            num_data_nodes = self.num_data_nodes

        graph["data"].x = torch.randn(num_data_nodes, self.n_input_data_features)
        if self.n_hidden_data_features > 0:
            graph["hidden"].x = torch.randn(
                graph["hidden"].num_nodes, self.n_hidden_data_features
            )
        else:
            graph["hidden"].x = torch.zeros(graph["hidden"].num_nodes, 0)
        graph["data"].y = torch.randn(num_data_nodes, self.n_output_data_features)
        return graph

    def collate_fn(self, graphs: list[HeteroData]) -> Batch:
        """Collate graphs into a batched HeteroData.

        Parameters
        ----------
        graphs : list[HeteroData]
            Graph samples.

        Returns
        -------
        Batch
            Batched graphs.
        """
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
        graph_builder: GraphBuilder,
        n_unique_graphs: int = 1,
    ):
        """Initialize the timeseries dummy dataset.

        Parameters
        ----------
        num_samples : int
            Number of samples in the dataset.
        num_data_nodes : int | dict[int, int]
            Number of data nodes per graph.
        n_state_features : int
            State feature dimension.
        n_forcing_features : int
            Forcing feature dimension.
        n_static_features : int
            Static feature dimension.
        ar_steps : int
            Autoregressive rollout length.
        n_hidden_data_features : int
            Hidden node feature dimension.
        graph_builder : GraphBuilder
            Graph builder used to create topology.
        n_unique_graphs : int, optional
            Number of unique graphs, by default 1.

        Returns
        -------
        None
        """
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_state_features = n_state_features
        self.n_forcing_features = n_forcing_features
        self.n_static_features = n_static_features
        self.ar_steps = ar_steps
        self.n_hidden_data_features = n_hidden_data_features
        self.graph_builder = graph_builder
        self.n_unique_graphs = n_unique_graphs

        self.graphs: list[HeteroData] = []
        for gid in range(n_unique_graphs):
            num_nodes = (
                num_data_nodes[gid]
                if isinstance(num_data_nodes, dict)
                else num_data_nodes
            )
            coords = _make_grid_coords(num_nodes)
            if coords.shape[0] != num_nodes:
                raise ValueError(
                    f"Generated coords has {coords.shape[0]} nodes but dataset expects {num_nodes}."
                )
            g = self.graph_builder(coords)
            g.graph_id = torch.tensor([gid], dtype=torch.long)
            self.graphs.append(g)

    def __len__(self) -> int:
        """Return dataset length.

        Returns
        -------
        int
            Number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> HeteroData:
        """Return a single timeseries graph sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        HeteroData
            Graph with autoregressive fields populated.
        """
        graph = self.graphs[idx % self.n_unique_graphs].clone()
        gid = int(graph.graph_id.item())
        num_nodes = (
            self.num_data_nodes[gid]
            if isinstance(self.num_data_nodes, dict)
            else self.num_data_nodes
        )

        graph["data"].x_init_states = torch.randn(num_nodes, self.n_state_features, 2)
        graph["data"].x_forcing = torch.randn(
            num_nodes, self.n_forcing_features, self.ar_steps
        )
        graph["data"].x_static = torch.randn(num_nodes, self.n_static_features)
        graph["data"].x = graph["data"].x_init_states[:, :, -1]
        graph["data"].y = torch.randn(
            num_nodes, self.n_state_features, self.ar_steps
        )  # [N, d_state, T]

        if self.n_hidden_data_features > 0:
            graph["hidden"].x = torch.randn(
                graph["hidden"].num_nodes, self.n_hidden_data_features
            )
        else:
            graph["hidden"].x = torch.zeros(graph["hidden"].num_nodes, 0)
        return graph

    def collate_fn(self, graphs: list[HeteroData]) -> Batch:
        """Collate graphs into a batched HeteroData.

        Parameters
        ----------
        graphs : list[HeteroData]
            Graph samples.

        Returns
        -------
        Batch
            Batched graphs.
        """
        return Batch.from_data_list(graphs)


class WeatherDuckDataModule(pl.LightningDataModule):
    """
    LightningDataModule providing dummy weather graphs via PyG DataLoader.
    """

    def __init__(
        self,
        graph_builder: GraphBuilder,
        num_samples: int = 128,
        num_data_nodes: int | dict[int, int] = 64,
        n_input_data_features: int = 8,
        n_output_data_features: int = 8,
        n_hidden_data_features: int = 0,
        batch_size: int = 4,
        n_unique_graphs: int = 1,
    ):
        """Initialize the dummy datamodule.

        Parameters
        ----------
        graph_builder : GraphBuilder
            Graph builder used to create topology.
        num_samples : int, optional
            Number of samples, by default 128.
        num_data_nodes : int | dict[int, int], optional
            Number of data nodes per graph.
        n_input_data_features : int, optional
            Input feature dimension.
        n_output_data_features : int, optional
            Output feature dimension.
        n_hidden_data_features : int, optional
            Hidden node feature dimension.
        batch_size : int, optional
            Batch size.
        n_unique_graphs : int, optional
            Number of unique graphs.

        Returns
        -------
        None
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_input_data_features = n_input_data_features
        self.n_output_data_features = n_output_data_features
        self.n_hidden_data_features = n_hidden_data_features
        self.graph_builder = graph_builder
        self.batch_size = batch_size
        self.n_unique_graphs = n_unique_graphs

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets for the requested stage.

        Parameters
        ----------
        stage : Optional[str], optional
            Lightning stage hint, by default None.

        Returns
        -------
        None
        """
        self.train_ds = DummyWeatherDataset(
            num_samples=self.num_samples,
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            n_hidden_data_features=self.n_hidden_data_features,
            graph_builder=self.graph_builder,
            n_unique_graphs=self.n_unique_graphs,
        )
        self.val_ds = DummyWeatherDataset(
            num_samples=max(8, self.num_samples // 10),
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            n_hidden_data_features=self.n_hidden_data_features,
            graph_builder=self.graph_builder,
            n_unique_graphs=self.n_unique_graphs,
        )
        self.test_ds = DummyWeatherDataset(
            num_samples=max(8, self.num_samples // 10),
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            n_hidden_data_features=self.n_hidden_data_features,
            graph_builder=self.graph_builder,
            n_unique_graphs=self.n_unique_graphs,
        )

    def train_dataloader(self) -> GeoDataLoader:
        """Return the training dataloader.

        Returns
        -------
        GeoDataLoader
            Training dataloader.
        """
        return GeoDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_ds.collate_fn,
        )

    def val_dataloader(self) -> GeoDataLoader:
        """Return the validation dataloader.

        Returns
        -------
        GeoDataLoader
            Validation dataloader.
        """
        return GeoDataLoader(
            self.val_ds, batch_size=self.batch_size, collate_fn=self.val_ds.collate_fn
        )

    def test_dataloader(self) -> GeoDataLoader:
        """Return the test dataloader.

        Returns
        -------
        GeoDataLoader
            Test dataloader.
        """
        return GeoDataLoader(
            self.test_ds, batch_size=self.batch_size, collate_fn=self.test_ds.collate_fn
        )


class TimeseriesWeatherDataModule(pl.LightningDataModule):
    """
    DataModule for timeseries dummy data compatible with AutoRegressiveForecaster.
    """

    def __init__(
        self,
        graph_builder: GraphBuilder,
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
        """Initialize the timeseries dummy datamodule.

        Parameters
        ----------
        graph_builder : GraphBuilder
            Graph builder used to create topology.
        num_samples : int, optional
            Number of samples, by default 128.
        num_data_nodes : int | dict[int, int], optional
            Number of data nodes per graph.
        n_state_features : int, optional
            State feature dimension.
        n_forcing_features : int, optional
            Forcing feature dimension.
        n_static_features : int, optional
            Static feature dimension.
        ar_steps : int, optional
            Autoregressive rollout length.
        n_hidden_data_features : int, optional
            Hidden node feature dimension.
        batch_size : int, optional
            Batch size.
        n_unique_graphs : int, optional
            Number of unique graphs.

        Returns
        -------
        None
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_state_features = n_state_features
        self.n_forcing_features = n_forcing_features
        self.n_static_features = n_static_features
        self.ar_steps = ar_steps
        self.n_hidden_data_features = n_hidden_data_features
        self.graph_builder = graph_builder
        self.batch_size = batch_size
        self.n_unique_graphs = n_unique_graphs

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets for the requested stage.

        Parameters
        ----------
        stage : Optional[str], optional
            Lightning stage hint, by default None.

        Returns
        -------
        None
        """
        self.train_ds = TimeseriesDummyWeatherDataset(
            num_samples=self.num_samples,
            num_data_nodes=self.num_data_nodes,
            n_state_features=self.n_state_features,
            n_forcing_features=self.n_forcing_features,
            n_static_features=self.n_static_features,
            ar_steps=self.ar_steps,
            n_hidden_data_features=self.n_hidden_data_features,
            graph_builder=self.graph_builder,
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
            graph_builder=self.graph_builder,
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
            graph_builder=self.graph_builder,
            n_unique_graphs=self.n_unique_graphs,
        )

    def train_dataloader(self) -> GeoDataLoader:
        """Return the training dataloader.

        Returns
        -------
        GeoDataLoader
            Training dataloader.
        """
        return GeoDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_ds.collate_fn,
        )

    def val_dataloader(self) -> GeoDataLoader:
        """Return the validation dataloader.

        Returns
        -------
        GeoDataLoader
            Validation dataloader.
        """
        return GeoDataLoader(
            self.val_ds, batch_size=self.batch_size, collate_fn=self.val_ds.collate_fn
        )

    def test_dataloader(self) -> GeoDataLoader:
        """Return the test dataloader.

        Returns
        -------
        GeoDataLoader
            Test dataloader.
        """
        return GeoDataLoader(
            self.test_ds, batch_size=self.batch_size, collate_fn=self.test_ds.collate_fn
        )
