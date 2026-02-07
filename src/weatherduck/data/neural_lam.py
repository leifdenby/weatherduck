from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from neural_lam.datastore.mdp import MDPDatastore
from neural_lam.weather_dataset import WeatherDataset
from torch.utils.data import Dataset
from torch_geometric.data import Batch, HeteroData

from ..graphs import GraphProvider
from .base import BaseWeatherDataModule

__all__ = ["MDPDataModule"]


class NeuralLamWeatherGraphDataset(Dataset):
    """WeatherDataset wrapper that yields WeatherDuck HeteroData."""

    def __init__(self, dataset: WeatherDataset, graph_builder: GraphProvider):
        """Initialize the wrapper dataset.

        Parameters
        ----------
        dataset : WeatherDataset
            Neural-lam dataset instance.
        graph_builder : GraphProvider
            Graph provider used to construct topology from coordinates.

        Returns
        -------
        None
        """
        self.dataset = dataset
        self.graph_builder = graph_builder
        coords = np.asarray(self.dataset.datastore.get_xy("state", stacked=True))
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Expected coords with shape [N, 2] from datastore.get_xy.")
        self.coords = coords
        domain_id = f"mdp:{self.dataset.datastore.root_path}"
        self.graph = self.graph_builder(domain_id=domain_id, coords=self.coords)
        if self.graph["data"].num_nodes != self.coords.shape[0]:
            raise ValueError(
                "Graph/data node count mismatch: "
                f"{self.graph['data'].num_nodes} != {self.coords.shape[0]}."
            )

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
            Graph populated with fields for autoregressive forecasting:

            ``graph["data"].x_init_states`` : ``[N_data, F_data, 2]``
            ``graph["data"].x_forcing`` : ``[N_data, F_forcing, T]``
            ``graph["data"].x_static`` : ``[N_data, F_static]``
            ``graph["data"].x`` : ``[N_data, F_data]``
            ``graph["data"].y`` : ``[N_data, F_data, T]``

            All edges and hidden node attributes are inherited from the base
            graph.

            where:
            - ``N_data``: number of data/grid nodes
            - ``F_data``: number of state features per node
            - ``F_forcing``: number of forcing features per node
            - ``F_static``: number of static features per node
            - ``T``: number of autoregressive target steps
        """
        sample = self.dataset[idx]
        if isinstance(sample, tuple):
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
        graph = self.graph.clone()

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
    graph_builder: GraphProvider
    ar_steps_train: int = 3
    ar_steps_eval: int = 25
    standardize: bool = True
    num_past_forcing_steps: int = 1
    num_future_forcing_steps: int = 1
    batch_size: int = 4
    num_workers: int = 0

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
        return NeuralLamWeatherGraphDataset(base, graph_builder=self.graph_builder)
