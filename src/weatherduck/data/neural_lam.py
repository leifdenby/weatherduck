from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from neural_lam.datastore.mdp import MDPDatastore
from neural_lam.weather_dataset import WeatherDataset
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torch_geometric.data import Batch, HeteroData

from .base import BaseWeatherDataModule
from .neural_lam_graph_data import build_graph_sizes, load_graph

__all__ = ["MDPDataModule"]


class WeatherDatasetWithGraph(Dataset):
    """Dataset wrapper that pairs weather samples with a static graph.

    Parameters
    ----------
    weather_dataset : WeatherDataset
        Dataset providing weather samples.
    graph_name : str
        Graph directory name under the datastore root.
    device : str, optional
        Device for loading graph tensors, by default "cpu".

    Returns
    -------
    None
    """

    def __init__(
        self,
        weather_dataset: WeatherDataset,
        graph_name: str,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.weather_dataset = weather_dataset
        self.graph_name = graph_name
        self.device = device
        self.datastore = weather_dataset.datastore

        self.graph_dir_path = Path(self.datastore.root_path) / "graph" / self.graph_name

        graph_edges_and_features = load_graph(
            graph_dir_path=self.graph_dir_path, device=self.device
        )
        self.graph_edges_and_features = graph_edges_and_features
        self.graph_sizes = build_graph_sizes(graph_edges_and_features)
        self.graph_payload = graph_edges_and_features.as_batch_dict()
        self.hierarchical = self.graph_sizes.hierarchical

    def __len__(self) -> int:
        """Return dataset length.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.weather_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a sample with attached static graph payload.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Dict[str, Any]
            Weather sample including "graph" payload.
        """
        sample = self.weather_dataset[idx]
        if isinstance(sample, dict):
            sample = sample.copy()
        elif isinstance(sample, tuple):
            if len(sample) != 4:
                raise ValueError(
                    "Expected tuple sample with 4 entries "
                    "(init_states, target_states, forcing_features, batch_times)."
                )
            sample = {
                "init_states": sample[0],
                "target_states": sample[1],
                "forcing_features": sample[2],
                "batch_times": sample[3],
            }
        else:
            raise TypeError(
                "Expected sample to be dict or tuple, got " f"{type(sample).__name__}."
            )
        sample["graph"] = copy.deepcopy(self.graph_payload)
        return sample

    @staticmethod
    def collate_fn(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function that keeps a single copy of the static graph.

        Parameters
        ----------
        batch : Iterable[Dict[str, Any]]
            Samples to collate.

        Returns
        -------
        Dict[str, Any]
            Collated batch with a shared "graph" entry.
        """
        batch = list(batch)
        if not batch:
            raise ValueError("Empty batch provided to collate_fn.")

        graphs = [entry.get("graph") for entry in batch]
        if any(graph is None for graph in graphs):
            raise ValueError("Graph entry missing from batch sample.")

        data_without_graph = [
            {key: value for key, value in entry.items() if key != "graph"}
            for entry in batch
        ]
        collated_data = default_collate(data_without_graph)
        collated_data["graph"] = graphs[0]
        return collated_data


def _graph_payload_to_heterodata(graph_payload: Dict[str, Any]) -> HeteroData:
    """Convert a neural-lam graph payload to HeteroData.

    Parameters
    ----------
    graph_payload : Dict[str, Any]
        Graph payload from WeatherDatasetWithGraph ("graph" entry).

    Returns
    -------
    HeteroData
        WeatherDuck-compatible graph with static edges/features.
    """
    if graph_payload.get("hierarchical"):
        raise ValueError(
            "Hierarchical neural-lam graphs are not supported in WeatherDuck yet."
        )

    graph = HeteroData()
    m2g_edge_index = graph_payload["m2g_edge_index"]
    g2m_edge_index = graph_payload["g2m_edge_index"]
    mesh_static = graph_payload["mesh_static_features"]
    num_hidden_nodes = mesh_static.shape[0]
    if m2g_edge_index.numel() == 0 or g2m_edge_index.numel() == 0:
        num_data_nodes = 0
    else:
        max_grid_idx = int(
            torch.max(torch.cat([g2m_edge_index[0], m2g_edge_index[1]], dim=0)).item()
        )
        num_data_nodes = max_grid_idx - num_hidden_nodes + 1
        if num_data_nodes < 0:
            raise ValueError("Computed negative data node count from graph indices.")
    graph["data"].x = torch.zeros(num_data_nodes, 0, device=m2g_edge_index.device)

    if mesh_static.shape[0] != num_hidden_nodes:
        raise ValueError(
            "Mesh static features do not match hidden node count: "
            f"{mesh_static.shape[0]} != {num_hidden_nodes}."
        )
    graph["hidden"].x = mesh_static

    g2m_edge = g2m_edge_index.clone()
    g2m_edge[0] = g2m_edge[0] - num_hidden_nodes
    graph["data", "to", "hidden"].edge_index = g2m_edge
    graph["data", "to", "hidden"].edge_attr = graph_payload["g2m_features"]

    m2g_edge = m2g_edge_index.clone()
    m2g_edge[1] = m2g_edge[1] - num_hidden_nodes
    graph["hidden", "to", "data"].edge_index = m2g_edge
    graph["hidden", "to", "data"].edge_attr = graph_payload["m2g_features"]

    graph["hidden", "to", "hidden"].edge_index = graph_payload["m2m_edge_index"]
    graph["hidden", "to", "hidden"].edge_attr = graph_payload["m2m_features"]

    return graph


class NeuralLamWeatherDataset(Dataset):
    """WeatherDatasetWithGraph wrapper that yields WeatherDuck HeteroData."""

    def __init__(self, dataset: WeatherDatasetWithGraph):
        """Initialize the wrapper dataset.

        Parameters
        ----------
        dataset : WeatherDatasetWithGraph
            Neural-lam dataset with graph payload.

        Returns
        -------
        None
        """
        self.dataset = dataset

    def __len__(self) -> int:
        """Return dataset length.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> HeteroData:
        """Return a WeatherDuck-formatted sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        HeteroData
            Graph populated with autoregressive fields.
        """
        sample = self.dataset[idx]
        graph = _graph_payload_to_heterodata(sample["graph"]).clone()

        init_states = sample["init_states"]  # [2, N, F]
        target_states = sample["target_states"]  # [T, N, F]
        forcing = sample["forcing_features"]  # [T, N, Ff]

        init_states = init_states.permute(1, 2, 0)
        target_states = target_states.permute(1, 2, 0)
        forcing = forcing.permute(1, 2, 0)

        num_nodes = init_states.shape[0]
        graph["data"].x_init_states = init_states
        graph["data"].x_forcing = forcing
        graph["data"].x_static = torch.zeros(num_nodes, 0, device=init_states.device)
        graph["data"].x = init_states[:, :, -1]
        graph["data"].y = target_states
        if graph["data"].num_nodes != num_nodes:
            raise ValueError(
                "Graph/data node count mismatch: "
                f"{graph['data'].num_nodes} != {num_nodes}."
            )
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


@dataclass
class MDPDataModule(BaseWeatherDataModule):
    """DataModule for neural-lam MDPDatastore-backed datasets."""

    config_path: str
    graph_name: str
    ar_steps_train: int = 3
    ar_steps_eval: int = 25
    standardize: bool = True
    num_past_forcing_steps: int = 1
    num_future_forcing_steps: int = 1
    batch_size: int = 4
    num_workers: int = 0
    graph_device: str = "cpu"

    def __post_init__(self) -> None:
        """Initialize the datamodule after dataclass creation.

        Returns
        -------
        None
        """
        super().__init__(batch_size=self.batch_size, num_workers=self.num_workers)
        self._datastore = MDPDatastore(self.config_path)

    def get_dataset(self, split: str) -> Dataset:
        """Return the dataset for the requested split.

        Parameters
        ----------
        split : str
            Dataset split name ("train", "val", or "test").

        Returns
        -------
        Dataset
            Neural-lam-backed dataset instance.
        """
        ar_steps = self.ar_steps_train if split == "train" else self.ar_steps_eval
        base = WeatherDataset(
            datastore=self._datastore,
            split=split,
            ar_steps=ar_steps,
            standardize=self.standardize,
            num_past_forcing_steps=self.num_past_forcing_steps,
            num_future_forcing_steps=self.num_future_forcing_steps,
        )
        with_graph = WeatherDatasetWithGraph(
            base,
            graph_name=self.graph_name,
            device=self.graph_device,
        )
        return NeuralLamWeatherDataset(with_graph)
