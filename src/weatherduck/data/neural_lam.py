from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from neural_lam.datastore.mdp import MDPDatastore
from neural_lam.weather_dataset import WeatherDataset, WeatherDatasetWithGraph
from torch.utils.data import Dataset
from torch_geometric.data import Batch, HeteroData

from .base import BaseWeatherDataModule

__all__ = ["MDPDataModule"]


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
    num_data_nodes = (
        int(torch.max(m2g_edge_index).item()) + 1 if m2g_edge_index.numel() > 0 else 0
    )
    graph["data"].x = torch.zeros(num_data_nodes, 0, device=m2g_edge_index.device)

    mesh_static = graph_payload["mesh_static_features"]
    graph["hidden"].x = mesh_static

    graph["data", "to", "hidden"].edge_index = graph_payload["g2m_edge_index"]
    graph["data", "to", "hidden"].edge_attr = graph_payload["g2m_features"]

    graph["hidden", "to", "data"].edge_index = graph_payload["m2g_edge_index"]
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
