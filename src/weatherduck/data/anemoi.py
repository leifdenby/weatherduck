from __future__ import annotations

from functools import cached_property
from typing import Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from anemoi.datasets import open_dataset
from anemoi.graphs.create import GraphBuilder
from anemoi.training.data.grid_indices import BaseGridIndices, FullGrid
from anemoi.training.utils.usable_indices import get_usable_indices
from anemoi.utils.dates import frequency_to_seconds
from einops import rearrange
from torch.utils.data import IterableDataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GeoDataLoader

__all__ = ["AnemoiNativeGridDataModule"]


class _CachedGraphBuilder:
    """Cache graph creation for reuse across datasets.

    Parameters
    ----------
    graph_builder : Any
        Graph builder that provides a create() method.
    """

    def __init__(self, graph_builder: GraphBuilder) -> None:
        self._graph_builder = graph_builder
        self._graph: HeteroData | None = None

    def create(self) -> HeteroData:
        """Build or return the cached graph.

        Returns
        -------
        HeteroData
            Cached graph instance.
        """
        if self._graph is None:
            self._graph = self._graph_builder.create()
        return self._graph


class UnshardedNativeGridDataset(IterableDataset):
    """
    Iterable dataset for Anemoi data without internal sharding.

    This mirrors Anemoi's NativeGridDataset but omits any worker or rank-based
    sharding. Any spatial selection should be handled by the graph or external
    grid indices configuration.
    """

    def __init__(
        self,
        data_reader,
        grid_indices: BaseGridIndices,
        relative_date_indices: list[int],
        *,
        timestep: str = "6h",
        shuffle: bool = True,
        label: str = "generic",
        seed: int = 0,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        data_reader : Any
            Dataset reader from anemoi.datasets.open_dataset.
        grid_indices : BaseGridIndices
            Spatial grid selection configuration.
        relative_date_indices : list[int]
            Relative time indices to sample.
        timestep : str, optional
            Dataset timestep, by default "6h".
        shuffle : bool, optional
            Shuffle samples each iteration, by default True.
        label : str, optional
            Dataset label for logging.
        seed : int, optional
            RNG seed used for shuffling.
        """
        super().__init__()
        self.label = label
        self.data = data_reader
        self.timestep = timestep
        self.grid_indices = grid_indices
        self.relative_date_indices = relative_date_indices
        self.shuffle = shuffle
        self.seed = seed

        self.ensemble_dim: int = 2
        self.ensemble_size = self.data.shape[self.ensemble_dim]

    @cached_property
    def statistics(self) -> dict:
        """Return dataset statistics.

        Returns
        -------
        dict
            Dataset statistics.
        """
        return self.data.statistics

    @cached_property
    def statistics_tendencies(self) -> dict:
        """Return dataset tendency statistics.

        Returns
        -------
        dict
            Tendency statistics (may be None).
        """
        try:
            return self.data.statistics_tendencies(self.timestep)
        except (KeyError, AttributeError):
            return None

    @cached_property
    def metadata(self) -> dict:
        """Return dataset metadata.

        Returns
        -------
        dict
            Dataset metadata.
        """
        return self.data.metadata()

    @cached_property
    def supporting_arrays(self) -> dict:
        """Return dataset supporting arrays.

        Returns
        -------
        dict
            Supporting arrays.
        """
        return self.data.supporting_arrays()

    @cached_property
    def name_to_index(self) -> dict:
        """Return variable name-to-index mapping.

        Returns
        -------
        dict
            Variable index mapping.
        """
        return self.data.name_to_index

    @cached_property
    def resolution(self) -> dict:
        """Return dataset resolution.

        Returns
        -------
        dict
            Resolution metadata.
        """
        return self.data.resolution

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        """Return valid date indices for sampling.

        Returns
        -------
        np.ndarray
            Valid date indices.
        """
        trajectory_ids = getattr(self.data, "trajectory_ids", None)
        return get_usable_indices(
            self.data.missing,
            len(self.data),
            np.array(self.relative_date_indices, dtype=np.int64),
            trajectory_ids,
        )

    def _grid_selection(self):
        if hasattr(self.grid_indices, "grid_indices"):
            return self.grid_indices.grid_indices
        return slice(None)

    def __iter__(self) -> torch.Tensor:
        """Return an iterator over unsharded samples.

        Returns
        -------
        Iterator[torch.Tensor]
            Samples shaped [time, ensemble, nodes, variables].
        """
        rng = np.random.default_rng(seed=self.seed)
        if self.shuffle:
            indices = rng.choice(
                self.valid_date_indices,
                size=len(self.valid_date_indices),
                replace=False,
            )
        else:
            indices = self.valid_date_indices

        grid_indices = self._grid_selection()

        for i in indices:
            start = i + self.relative_date_indices[0]
            end = i + self.relative_date_indices[-1] + 1
            timeincrement = (
                self.relative_date_indices[1] - self.relative_date_indices[0]
            )
            if isinstance(grid_indices, slice):
                x = self.data[start:end:timeincrement, :, :, grid_indices]
            else:
                x = self.data[start:end:timeincrement, :, :, :]
                x = x[..., grid_indices]
            x = rearrange(
                x,
                "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
            )
            self.ensemble_dim = 1
            yield torch.from_numpy(x)

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Dataset: {self.data}
            Relative dates: {self.relative_date_indices}
        """


class _NativeGridGraphDataset(IterableDataset):
    def __init__(
        self,
        *,
        base_dataset: IterableDataset,
        graph_builder: GraphBuilder,
        name_to_index: dict[str, int],
        input_time_index: int,
        target_time_index: int,
        ensemble_index: int,
        data_node_type: str,
        hidden_node_type: str,
        append_graph_node_features: bool,
        use_hidden_node_features: bool,
        input_variable_names: Sequence[str] | None,
        output_variable_names: Sequence[str] | None,
        ar_steps: int | None,
    ) -> None:
        """Initialize the graph-wrapping dataset.

        Parameters
        ----------
        base_dataset : IterableDataset
            NativeGridDataset-like iterable yielding tensors shaped
            [time, ensemble, nodes, variables].
        graph_builder : GraphBuilder
            Graph builder used to construct the template graph.
        name_to_index : dict[str, int]
            Mapping from variable names to column indices.
        input_time_index : int
            Time index used for input features.
        target_time_index : int
            Time index used for target features.
        ensemble_index : int
            Ensemble member index to select.
        data_node_type : str
            Node type used for input/output data ("data" by default).
        hidden_node_type : str
            Hidden node type used by the model ("hidden" by default).
        append_graph_node_features : bool
            If True, concatenate graph node features to the input features.
        use_hidden_node_features : bool
            If False, replace hidden node features with an empty tensor.
        input_variable_names : Sequence[str] | None
            Optional variable names to select for input features.
        output_variable_names : Sequence[str] | None
            Optional variable names to select for target features.
        ar_steps : int | None
            If provided, emit autoregressive fields with this rollout length.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.graph_data = graph_builder.create()
        self.name_to_index = name_to_index
        self.input_time_index = input_time_index
        self.target_time_index = target_time_index
        self.ensemble_index = ensemble_index
        self.data_node_type = data_node_type
        self.hidden_node_type = hidden_node_type
        self.append_graph_node_features = append_graph_node_features
        self.use_hidden_node_features = use_hidden_node_features
        self.input_variable_names = input_variable_names
        self.output_variable_names = output_variable_names
        self.ar_steps = ar_steps

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Forward worker initialization to the wrapped dataset.

        Parameters
        ----------
        n_workers : int
            Total number of workers.
        worker_id : int
            Worker index.
        """
        if hasattr(self.base_dataset, "per_worker_init"):
            self.base_dataset.per_worker_init(n_workers=n_workers, worker_id=worker_id)

    def set_comm_group_info(self, *args, **kwargs) -> None:
        """Forward distributed group metadata to the wrapped dataset.

        Parameters
        ----------
        *args
            Positional arguments forwarded as-is.
        **kwargs
            Keyword arguments forwarded as-is.
        """
        if hasattr(self.base_dataset, "set_comm_group_info"):
            self.base_dataset.set_comm_group_info(*args, **kwargs)

    def _resolve_variable_indices(
        self, variable_names: Sequence[str] | None
    ) -> torch.Tensor:
        """Resolve variable indices from names or select all.

        Parameters
        ----------
        variable_names : Sequence[str] | None
            Variable names to resolve. If None, select all variables.

        Returns
        -------
        torch.Tensor
            Sorted variable indices as a 1D tensor.
        """
        if variable_names is None:
            return torch.tensor(sorted(self.name_to_index.values()), dtype=torch.long)
        indices = [self.name_to_index[name] for name in variable_names]
        return torch.tensor(sorted(indices), dtype=torch.long)

    def __iter__(self):
        """Yield HeteroData graphs with populated data.x/data.y.

        Returns
        -------
        Iterator[HeteroData]
            Iterator over graphs compatible with WeatherDuck models.
        """
        input_vars = self._resolve_variable_indices(self.input_variable_names)
        output_vars = self._resolve_variable_indices(self.output_variable_names)
        for sample in self.base_dataset:
            if sample.dim() != 4:
                raise ValueError(
                    "Expected Anemoi dataset sample to have shape [time, ensemble, nodes, variables]."
                )
            if self.ar_steps is None:
                input_slice = sample[self.input_time_index, self.ensemble_index]
                target_slice = sample[self.target_time_index, self.ensemble_index]
                data_x = input_slice[:, input_vars]
                data_y = target_slice[:, output_vars]
            else:
                required_steps = 2 + self.ar_steps
                if sample.shape[0] < required_steps:
                    raise ValueError(
                        "Autoregressive mode requires at least "
                        f"{required_steps} time steps, got {sample.shape[0]}."
                    )
                time_slice = sample[:required_steps, self.ensemble_index]
                init_states = time_slice[:2, :, input_vars]  # [2, N, F]
                target_seq = time_slice[2:, :, output_vars]  # [T, N, F]
                data_x = init_states[-1]
                data_y = target_seq.permute(1, 2, 0)

            graph = self.graph_data.clone()
            if graph[self.data_node_type].num_nodes != data_x.shape[0]:
                raise ValueError(
                    f"Data node count mismatch ({graph[self.data_node_type].num_nodes} != {data_x.shape[0]}). "
                    "Ensure grid_indices keeps the full grid or update the graph accordingly."
                )
            if self.append_graph_node_features:
                base_feats = graph[self.data_node_type].x.to(data_x.device)
                data_x = torch.cat([base_feats, data_x], dim=-1)
            graph[self.data_node_type].x = data_x
            graph[self.data_node_type].y = data_y
            if self.ar_steps is not None:
                graph[self.data_node_type].x_init_states = init_states.permute(1, 2, 0)
                graph[self.data_node_type].x_forcing = torch.zeros(
                    data_x.shape[0], 0, self.ar_steps, device=data_x.device
                )
                graph[self.data_node_type].x_static = torch.zeros(
                    data_x.shape[0], 0, device=data_x.device
                )

            if not self.use_hidden_node_features:
                num_hidden = graph[self.hidden_node_type].num_nodes
                graph[self.hidden_node_type].x = torch.zeros(
                    num_hidden, 0, device=data_x.device
                )
            graph.graph_id = torch.tensor([0], dtype=torch.long, device=data_x.device)
            yield graph


class AnemoiNativeGridDataModule(pl.LightningDataModule):
    """
    LightningDataModule built directly on UnshardedNativeGridDataset.

    This avoids requiring a full Anemoi BaseSchema; pass dataset configs and
    a graph, and the module will construct UnshardedNativeGridDataset instances
    and convert samples into WeatherDuck-ready HeteroData graphs.
    """

    def __init__(
        self,
        *,
        graph_builder: GraphBuilder,
        training: dict,
        validation: dict,
        test: dict,
        data_frequency: str,
        data_timestep: str,
        multistep_input: int = 1,
        rollout: int = 1,
        validation_rollout: int = 1,
        relative_date_indices: Optional[Sequence[int]] = None,
        grid_indices: Optional[BaseGridIndices] = None,
        reader_group_size: int = 1,
        grid_nodes_name: str = "data",
        batch_size: Optional[dict[str, int]] = None,
        num_workers: Optional[dict[str, int]] = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        input_time_index: int = 0,
        target_time_index: int = -1,
        ensemble_index: int = 0,
        data_node_type: str = "data",
        hidden_node_type: str = "hidden",
        append_graph_node_features: bool = False,
        use_hidden_node_features: bool = True,
        input_variable_names: Optional[Sequence[str]] = None,
        output_variable_names: Optional[Sequence[str]] = None,
        ar_steps: Optional[int] = None,
    ) -> None:
        """Initialize the NativeGrid-backed datamodule.

        Parameters
        ----------
        graph_builder : GraphBuilder
            Graph builder used to construct WeatherDuck batches.
        training : dict
            Anemoi dataset config for training split.
        validation : dict
            Anemoi dataset config for validation split.
        test : dict
            Anemoi dataset config for test split.
        data_frequency : str
            Data frequency (e.g., "6h").
        data_timestep : str
            Model timestep (e.g., "6h").
        multistep_input : int, optional
            Number of input steps, by default 1.
        rollout : int, optional
            Rollout length for training, by default 1.
        validation_rollout : int, optional
            Rollout length for validation, by default 1.
        relative_date_indices : Optional[Sequence[int]], optional
            Explicit relative date indices; overrides multistep/rollout.
        grid_indices : Optional[BaseGridIndices], optional
            Custom grid indices strategy; defaults to FullGrid.
        reader_group_size : int, optional
            Reader group size for grid sharding, by default 1.
        grid_nodes_name : str, optional
            Node type name in the graph for grid nodes.
        batch_size : Optional[dict[str, int]], optional
            Per-stage batch sizes.
        num_workers : Optional[dict[str, int]], optional
            Per-stage worker counts.
        pin_memory : bool, optional
            Enable pinned memory.
        prefetch_factor : int, optional
            Prefetch factor for workers.
        input_time_index : int, optional
            Time index used for input features.
        target_time_index : int, optional
            Time index used for target features.
        ensemble_index : int, optional
            Ensemble member index.
        data_node_type : str, optional
            Node type used for input/output data.
        hidden_node_type : str, optional
            Hidden node type used by the model.
        append_graph_node_features : bool, optional
            If True, concatenate graph node features to input data.
        use_hidden_node_features : bool, optional
            If False, replace hidden node features with an empty tensor.
        input_variable_names : Optional[Sequence[str]], optional
            Variable names to select for input features.
        output_variable_names : Optional[Sequence[str]], optional
            Variable names to select for target features.
        ar_steps : Optional[int], optional
            If set, emit autoregressive fields with this rollout length.
        """
        super().__init__()
        self.graph_builder = _CachedGraphBuilder(graph_builder)
        self.training_cfg = training
        self.validation_cfg = validation
        self.test_cfg = test
        self.data_frequency = data_frequency
        self.data_timestep = data_timestep
        self.multistep_input = multistep_input
        self.rollout = rollout
        self.validation_rollout = validation_rollout
        self.relative_date_indices = (
            list(relative_date_indices) if relative_date_indices is not None else None
        )
        self.grid_indices = grid_indices
        self.reader_group_size = reader_group_size
        self.grid_nodes_name = grid_nodes_name
        self.batch_size = batch_size or {
            "training": 2,
            "validation": 4,
            "test": 4,
        }
        self.num_workers = num_workers or {
            "training": 8,
            "validation": 8,
            "test": 8,
        }
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.input_time_index = input_time_index
        self.target_time_index = target_time_index
        self.ensemble_index = ensemble_index
        self.data_node_type = data_node_type
        self.hidden_node_type = hidden_node_type
        self.append_graph_node_features = append_graph_node_features
        self.use_hidden_node_features = use_hidden_node_features
        self.input_variable_names = input_variable_names
        self.output_variable_names = output_variable_names
        self.ar_steps = ar_steps

    @property
    def _graph_data(self) -> HeteroData:
        """Return the cached graph data built from the graph builder.

        Returns
        -------
        HeteroData
            Cached graph instance.
        """
        return self.graph_builder.create()

    @property
    def statistics(self) -> dict:
        """Return training dataset statistics.

        Returns
        -------
        dict
            Dataset statistics.
        """
        return self.ds_train.statistics

    @property
    def statistics_tendencies(self) -> dict:
        """Return training dataset tendency statistics.

        Returns
        -------
        dict
            Tendency statistics (may be None).
        """
        return self.ds_train.statistics_tendencies

    @property
    def metadata(self) -> dict:
        """Return training dataset metadata.

        Returns
        -------
        dict
            Dataset metadata.
        """
        return self.ds_train.metadata

    @property
    def supporting_arrays(self) -> dict:
        """Return supporting arrays for the dataset and grid indices.

        Returns
        -------
        dict
            Supporting arrays.
        """
        return self.ds_train.supporting_arrays | self._grid_indices.supporting_arrays

    def _timeincrement(self) -> int:
        """Compute time increment in data-frequency steps.

        Returns
        -------
        int
            Time increment in units of data frequency.
        """
        frequency = frequency_to_seconds(self.data_frequency)
        timestep = frequency_to_seconds(self.data_timestep)
        if timestep % frequency != 0:
            raise ValueError(
                f"Timestep {self.data_timestep} is not a multiple of frequency {self.data_frequency}."
            )
        return timestep // frequency

    def _relative_date_indices(self, val_rollout: int) -> list[int]:
        """Compute relative date indices for sampling.

        Parameters
        ----------
        val_rollout : int
            Rollout length used for validation.

        Returns
        -------
        list[int]
            Relative date indices for sampling.
        """
        if self.relative_date_indices is not None:
            return list(self.relative_date_indices)
        if self.ar_steps is not None:
            rollout = max(self.ar_steps, val_rollout)
            required_steps = 2 + rollout
        else:
            rollout = max(self.rollout, val_rollout)
            required_steps = self.multistep_input + rollout
        timeincrement = self._timeincrement()
        return [timeincrement * step for step in range(required_steps)]

    @property
    def _grid_indices(self) -> BaseGridIndices:
        """Return configured grid indices after setup.

        Returns
        -------
        BaseGridIndices
            Configured grid indices instance.
        """
        if self.grid_indices is None:
            grid_indices = FullGrid(
                nodes_name=self.grid_nodes_name,
                reader_group_size=self.reader_group_size,
            )
        else:
            grid_indices = self.grid_indices
        grid_indices.setup(self._graph_data)
        return grid_indices

    def _get_dataset(
        self,
        data_reader,
        *,
        shuffle: bool = True,
        val_rollout: int = 1,
        label: str = "generic",
    ) -> UnshardedNativeGridDataset:
        """Build an unsharded grid dataset for a reader.

        Parameters
        ----------
        data_reader : Any
            Dataset reader from anemoi.datasets.open_dataset.
        shuffle : bool, optional
            Shuffle samples, by default True.
        val_rollout : int, optional
            Rollout length used to compute relative dates, by default 1.
        label : str, optional
            Dataset label for logging.

        Returns
        -------
        UnshardedNativeGridDataset
            Iterable dataset for unsharded grid data.
        """
        if not hasattr(data_reader, "trajectory_ids"):
            data_reader.trajectory_ids = None
        return UnshardedNativeGridDataset(
            data_reader=data_reader,
            relative_date_indices=self._relative_date_indices(val_rollout),
            timestep=self.data_timestep,
            shuffle=shuffle,
            grid_indices=self._grid_indices,
            label=label,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize datasets for the requested stage.

        Parameters
        ----------
        stage : Optional[str], optional
            Lightning stage hint (unused), by default None.
        """
        train_reader = open_dataset(self.training_cfg)
        val_reader = open_dataset(self.validation_cfg)
        test_reader = open_dataset(self.test_cfg)

        self.ds_train = self._get_dataset(train_reader, label="train")
        self.ds_valid = self._get_dataset(
            val_reader,
            shuffle=False,
            val_rollout=self.validation_rollout,
            label="validation",
        )
        self.ds_test = self._get_dataset(test_reader, shuffle=False, label="test")

        self.train_ds = _NativeGridGraphDataset(
            base_dataset=self.ds_train,
            graph_builder=self.graph_builder,
            name_to_index=self.ds_train.name_to_index,
            input_time_index=self.input_time_index,
            target_time_index=self.target_time_index,
            ensemble_index=self.ensemble_index,
            data_node_type=self.data_node_type,
            hidden_node_type=self.hidden_node_type,
            append_graph_node_features=self.append_graph_node_features,
            use_hidden_node_features=self.use_hidden_node_features,
            input_variable_names=self.input_variable_names,
            output_variable_names=self.output_variable_names,
            ar_steps=self.ar_steps,
        )
        self.val_ds = _NativeGridGraphDataset(
            base_dataset=self.ds_valid,
            graph_builder=self.graph_builder,
            name_to_index=self.ds_train.name_to_index,
            input_time_index=self.input_time_index,
            target_time_index=self.target_time_index,
            ensemble_index=self.ensemble_index,
            data_node_type=self.data_node_type,
            hidden_node_type=self.hidden_node_type,
            append_graph_node_features=self.append_graph_node_features,
            use_hidden_node_features=self.use_hidden_node_features,
            input_variable_names=self.input_variable_names,
            output_variable_names=self.output_variable_names,
            ar_steps=self.ar_steps,
        )
        self.test_ds = _NativeGridGraphDataset(
            base_dataset=self.ds_test,
            graph_builder=self.graph_builder,
            name_to_index=self.ds_train.name_to_index,
            input_time_index=self.input_time_index,
            target_time_index=self.target_time_index,
            ensemble_index=self.ensemble_index,
            data_node_type=self.data_node_type,
            hidden_node_type=self.hidden_node_type,
            append_graph_node_features=self.append_graph_node_features,
            use_hidden_node_features=self.use_hidden_node_features,
            input_variable_names=self.input_variable_names,
            output_variable_names=self.output_variable_names,
            ar_steps=self.ar_steps,
        )

    def _get_dataloader(self, ds: IterableDataset, stage: str) -> GeoDataLoader:
        """Build a PyG dataloader for a stage.

        Parameters
        ----------
        ds : IterableDataset
            Dataset for the stage.
        stage : str
            One of "training", "validation", or "test".

        Returns
        -------
        GeoDataLoader
            PyG data loader.
        """
        assert stage in {"training", "validation", "test"}
        num_workers = self.num_workers.get(stage, 1)
        prefetch_factor = self.prefetch_factor if num_workers > 0 else None
        return GeoDataLoader(
            ds,
            batch_size=self.batch_size[stage],
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0,
        )

    def train_dataloader(self) -> GeoDataLoader:
        """Return the training dataloader.

        Returns
        -------
        GeoDataLoader
            Training loader.
        """
        return self._get_dataloader(self.train_ds, "training")

    def val_dataloader(self) -> GeoDataLoader:
        """Return the validation dataloader.

        Returns
        -------
        GeoDataLoader
            Validation loader.
        """
        return self._get_dataloader(self.val_ds, "validation")

    def test_dataloader(self) -> GeoDataLoader:
        """Return the test dataloader.

        Returns
        -------
        GeoDataLoader
            Test loader.
        """
        return self._get_dataloader(self.test_ds, "test")
