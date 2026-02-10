import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, HeteroData

from ..graphs import GraphProvider
from .base import BaseWeatherDataModule


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


def _populate_dummy_sample(
    graph: HeteroData,
    num_data_nodes: int,
    n_input_data_features: int,
    n_output_data_features: int,
) -> HeteroData:
    """Populate a graph by appending input/hidden features and adding targets.

    This appends ``n_input_data_features`` random input features to
    ``graph["data"].x``. It also sets ``graph["data"].y`` to random targets with
    ``n_output_data_features`` channels. Hidden node features are left unchanged.

    Parameters
    ----------
    graph : HeteroData
        Graph to augment. Expected structure:
        - Node types: ``data`` and ``hidden``.
        - ``graph["data"].x`` with shape ``[N_data, F_data]`` (will be appended).
        - ``graph["hidden"].x`` with shape ``[N_hidden, F_hidden]`` (will be appended).
        - Edge types:
          - ``("data","to","hidden")`` with ``edge_index`` shape ``[2, E_dh]``.
          - ``("hidden","to","hidden")`` with ``edge_index`` shape ``[2, E_hh]``.
          - ``("hidden","to","data")`` with ``edge_index`` shape ``[2, E_hd]``.
    num_data_nodes : int
        Number of data nodes to populate.
    n_input_data_features : int
        Input feature dimension for data nodes.
    n_output_data_features : int
        Output feature dimension for targets.
    Returns
    -------
    HeteroData
        Graph with randomized node features and targets appended. This function:
        - Concatenates random data-node features to ``graph["data"].x``.
        - Sets ``graph["data"].y`` to random targets with shape
          ``[N_data, n_output_data_features]``.
    """
    if n_input_data_features > 0:
        data_append = torch.randn(num_data_nodes, n_input_data_features)
    else:
        data_append = torch.zeros(num_data_nodes, 0)
    if "x" in graph["data"]:
        graph["data"].x = torch.cat([graph["data"].x, data_append], dim=-1)
    else:
        graph["data"].x = data_append

    graph["data"].y = torch.randn(num_data_nodes, n_output_data_features)
    return graph


def _populate_dummy_timeseries_sample(
    graph: HeteroData,
    num_data_nodes: int,
    n_state_features: int,
    n_forcing_features: int,
    n_static_features: int,
    ar_steps: int,
) -> HeteroData:
    """Append random time-series features and targets to an existing graph.

    Parameters
    ----------
    graph : HeteroData
        Graph to augment. Expected structure:
        - Node types: ``data`` and ``hidden``.
        - ``graph["data"].x`` with shape ``[N_data, F_static]`` holding static
          graph features (e.g. spatial coordinates); these are folded into
          ``x_static`` below.
        - ``graph["hidden"].x`` with shape ``[N_hidden, F_hidden]`` (may be present).
        - Edge types:
          - ``("data","to","hidden")`` with ``edge_index`` shape ``[2, E_dh]``.
          - ``("hidden","to","hidden")`` with ``edge_index`` shape ``[2, E_hh]``.
          - ``("hidden","to","data")`` with ``edge_index`` shape ``[2, E_hd]``.
    num_data_nodes : int
        Number of data nodes to populate.
    n_state_features : int
        State feature dimension.
    n_forcing_features : int
        Forcing feature dimension.
    n_static_features : int
        Number of additional static features to append beyond the static graph
        features already present in ``graph["data"].x``.
    ar_steps : int
        Autoregressive rollout length.
    Returns
    -------
    HeteroData
        Graph with autoregressive fields populated. This function:
        - Sets ``graph["data"].x_init_states`` to shape ``[N, d_state, 2]``.
        - Sets ``graph["data"].x_forcing`` to shape ``[N, d_forcing, T]``.
        - Sets ``graph["data"].x_static`` by concatenating ``graph["data"].x``
          with ``n_static_features`` extra static features.
        - Sets ``graph["data"].y`` to shape ``[N, d_state, T]``.
        - Does not modify ``graph["hidden"].x`` (hidden features remain static).
    """
    graph["data"].x_init_states = torch.randn(num_data_nodes, n_state_features, 2)
    graph["data"].x_forcing = torch.randn(num_data_nodes, n_forcing_features, ar_steps)
    base_static = (
        graph["data"].x if "x" in graph["data"] else torch.zeros(num_data_nodes, 0)
    )
    extra_static = torch.randn(num_data_nodes, n_static_features)
    graph["data"].x_static = torch.cat([base_static, extra_static], dim=-1)
    if "x" in graph["data"]:
        del graph["data"].x
    graph["data"].y = torch.randn(num_data_nodes, n_state_features, ar_steps)
    graph["data"].num_nodes = num_data_nodes

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
        graph_provider: GraphProvider,
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
        graph_provider : GraphProvider
            Graph provider used to create topology.
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
        self.graph_provider = graph_provider
        self.n_unique_graphs = n_unique_graphs
        self._domain_coords: dict[int, np.ndarray] = {}
        for gid in range(n_unique_graphs):
            if isinstance(self.num_data_nodes, dict):
                num_nodes = self.num_data_nodes.get(gid)
                assert (
                    num_nodes is not None
                ), f"num_data_nodes missing entry for graph id {gid}"
            else:
                num_nodes = self.num_data_nodes
            coords = _make_grid_coords(num_nodes)
            if coords.shape[0] != num_nodes:
                raise ValueError(
                    f"Generated coords has {coords.shape[0]} nodes but dataset expects {num_nodes}."
                )
            self._domain_coords[gid] = coords

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
        gid = idx % self.n_unique_graphs
        if isinstance(self.num_data_nodes, dict):
            num_data_nodes = self.num_data_nodes.get(gid)
            assert (
                num_data_nodes is not None
            ), f"num_data_nodes missing entry for graph id {gid}"
        else:
            num_data_nodes = self.num_data_nodes

        coords = self._domain_coords[gid]
        domain_id = f"dummy-{gid}"
        graph = self.graph_provider(domain_id=domain_id, coords=coords)
        graph.graph_id = torch.tensor([gid], dtype=torch.long)
        return _populate_dummy_sample(
            graph=graph,
            num_data_nodes=num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
        )

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
        graph_provider: GraphProvider,
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
        graph_provider : GraphProvider
            Graph provider used to create topology.
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
        self.graph_provider = graph_provider
        self.n_unique_graphs = n_unique_graphs
        self._domain_coords: dict[int, np.ndarray] = {}
        for gid in range(n_unique_graphs):
            if isinstance(self.num_data_nodes, dict):
                num_nodes = self.num_data_nodes.get(gid)
                assert (
                    num_nodes is not None
                ), f"num_data_nodes missing entry for graph id {gid}"
            else:
                num_nodes = self.num_data_nodes
            coords = _make_grid_coords(num_nodes)
            if coords.shape[0] != num_nodes:
                raise ValueError(
                    f"Generated coords has {coords.shape[0]} nodes but dataset expects {num_nodes}."
                )
            self._domain_coords[gid] = coords

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
            Graph with autoregressive fields populated:
            - ``graph["data"].x_init_states`` with shape ``[N, d_state, 2]``.
            - ``graph["data"].x_forcing`` with shape ``[N, d_forcing, T]``.
            - ``graph["data"].x_static`` with shape ``[N, d_static]``.
            - ``graph["data"].y`` with shape ``[N, d_state, T]``.
            - ``graph["hidden"].x`` left unchanged.
        """
        gid = idx % self.n_unique_graphs
        num_nodes = (
            self.num_data_nodes[gid]
            if isinstance(self.num_data_nodes, dict)
            else self.num_data_nodes
        )
        coords = self._domain_coords[gid]
        domain_id = f"dummy-timeseries-{gid}"
        graph = self.graph_provider(domain_id=domain_id, coords=coords)
        graph.graph_id = torch.tensor([gid], dtype=torch.long)

        graph = _populate_dummy_timeseries_sample(
            graph=graph,
            num_data_nodes=num_nodes,
            n_state_features=self.n_state_features,
            n_forcing_features=self.n_forcing_features,
            n_static_features=self.n_static_features,
            ar_steps=self.ar_steps,
        )
        # Ensure autoregressive inputs don't carry a precomputed data.x
        assert "x" not in graph["data"]
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


class DummyWeatherDataModule(BaseWeatherDataModule):
    """
    LightningDataModule providing dummy weather graphs via PyG DataLoader.
    """

    def __init__(
        self,
        graph_provider: GraphProvider,
        num_samples: int = 128,
        num_data_nodes: int | dict[int, int] = 64,
        n_input_data_features: int = 8,
        n_output_data_features: int = 8,
        batch_size: int = 4,
        n_unique_graphs: int = 1,
    ):
        """Initialize the dummy datamodule.

        Parameters
        ----------
        graph_provider : GraphProvider
            Graph provider used to create topology.
        num_samples : int, optional
            Number of samples, by default 128.
        num_data_nodes : int | dict[int, int], optional
            Number of data nodes per graph.
        n_input_data_features : int, optional
            Input feature dimension.
        n_output_data_features : int, optional
            Output feature dimension.
        batch_size : int, optional
            Batch size.
        n_unique_graphs : int, optional
            Number of unique graphs.

        Returns
        -------
        None
        """
        super().__init__(batch_size=batch_size)
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_input_data_features = n_input_data_features
        self.n_output_data_features = n_output_data_features
        self.graph_provider = graph_provider
        self.n_unique_graphs = n_unique_graphs

    def get_dataset(self, split: str) -> Dataset:
        """Return the dataset for the requested split.

        Parameters
        ----------
        split : str
            Dataset split name ("train", "val", or "test").

        Returns
        -------
        Dataset
            Dummy dataset instance.
        """
        num_samples = (
            self.num_samples if split == "train" else max(8, self.num_samples // 10)
        )
        return DummyWeatherDataset(
            num_samples=num_samples,
            num_data_nodes=self.num_data_nodes,
            n_input_data_features=self.n_input_data_features,
            n_output_data_features=self.n_output_data_features,
            graph_provider=self.graph_provider,
            n_unique_graphs=self.n_unique_graphs,
        )


class DummyTimeseriesWeatherDataModule(BaseWeatherDataModule):
    """
    DataModule for timeseries dummy data compatible with AutoRegressiveForecaster.
    """

    def __init__(
        self,
        graph_provider: GraphProvider,
        num_samples: int = 128,
        num_data_nodes: int | dict[int, int] = 64,
        n_state_features: int = 4,
        n_forcing_features: int = 2,
        n_static_features: int = 1,
        ar_steps: int = 3,
        batch_size: int = 4,
        n_unique_graphs: int = 1,
    ):
        """Initialize the timeseries dummy datamodule.

        Parameters
        ----------
        graph_provider : GraphProvider
            Graph provider used to create topology.
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
        batch_size : int, optional
            Batch size.
        n_unique_graphs : int, optional
            Number of unique graphs.

        Returns
        -------
        None
        """
        super().__init__(batch_size=batch_size)
        self.num_samples = num_samples
        self.num_data_nodes = num_data_nodes
        self.n_state_features = n_state_features
        self.n_forcing_features = n_forcing_features
        self.n_static_features = n_static_features
        self.ar_steps = ar_steps
        self.graph_provider = graph_provider
        self.n_unique_graphs = n_unique_graphs

    def get_dataset(self, split: str) -> Dataset:
        """Return the dataset for the requested split.

        Parameters
        ----------
        split : str
            Dataset split name ("train", "val", or "test").

        Returns
        -------
        Dataset
            Timeseries dummy dataset instance.
        """
        num_samples = (
            self.num_samples if split == "train" else max(8, self.num_samples // 10)
        )
        return TimeseriesDummyWeatherDataset(
            num_samples=num_samples,
            num_data_nodes=self.num_data_nodes,
            n_state_features=self.n_state_features,
            n_forcing_features=self.n_forcing_features,
            n_static_features=self.n_static_features,
            ar_steps=self.ar_steps,
            graph_provider=self.graph_provider,
            n_unique_graphs=self.n_unique_graphs,
        )
