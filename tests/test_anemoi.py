import os
import time
from dataclasses import dataclass
from pathlib import Path

import fiddle as fdl
import pytest
import pytorch_lightning as pl
import torch
from anemoi.graphs.edges import CutOffEdges, KNNEdges, MultiScaleEdges
from anemoi.graphs.edges.attributes import EdgeDirection, EdgeLength
from anemoi.graphs.nodes import AnemoiDatasetNodes, LimitedAreaTriNodes
from anemoi.graphs.nodes.attributes import CutOutMask
from loguru import logger
from torch import nn
from torch_geometric.data import HeteroData

from weatherduck import AutoRegressiveForecaster, build_encode_process_decode_model
from weatherduck.data.anemoi import AnemoiNativeGridDataModule


def _load_or_build_graph(
    cache_path: str | None,
    builder,
    *,
    label: str,
) -> HeteroData:
    """
    Load a cached graph or build and cache it.

    Parameters
    ----------
    cache_path : str | None
        Path to cache file. If None, always build.
    builder : Callable[[], HeteroData]
        Function that builds the graph when called.
    label : str
        Label for timing output.

    Returns
    -------
    HeteroData
        Graph instance.
    """
    if cache_path:
        logger.info(f"Using graph cache at: {cache_path}")
        graph_cache_path = Path(cache_path)
        if graph_cache_path.exists():
            logger.info(f"Loading graph from cache: {graph_cache_path}")
            start = time.perf_counter()
            graph = torch.load(graph_cache_path, weights_only=False)
            print(f"[timing] load_{label}: {time.perf_counter() - start:.2f}s")
            return graph
        start = time.perf_counter()
        graph = builder()
        print(f"[timing] build_{label}: {time.perf_counter() - start:.2f}s")
        graph_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, graph_cache_path)
        return graph

    start = time.perf_counter()
    graph = builder()
    print(f"[timing] build_{label}: {time.perf_counter() - start:.2f}s")
    return graph


@dataclass
class AnemoiGraph:
    data_nodes: AnemoiDatasetNodes
    hidden_nodes: LimitedAreaTriNodes
    cutout_mask: CutOutMask
    edge_length: EdgeLength
    edge_dirs: EdgeDirection
    dh_edges: CutOffEdges
    hh_edges: MultiScaleEdges
    hd_edges: KNNEdges

    def build(self) -> HeteroData:
        graph = HeteroData()
        graph = self.data_nodes.register_nodes(graph)
        graph["data"]["cutout_mask"] = self.cutout_mask.compute(graph, "data")

        graph = self.hidden_nodes.register_nodes(graph)

        graph = self.dh_edges.update_graph(graph, attrs_config=None)
        graph["data", "to", "hidden"]["edge_length"] = self.edge_length(
            x=(graph["data"], graph["hidden"]),
            edge_index=graph["data", "to", "hidden"].edge_index,
        )
        graph["data", "to", "hidden"]["edge_dirs"] = self.edge_dirs(
            x=(graph["data"], graph["hidden"]),
            edge_index=graph["data", "to", "hidden"].edge_index,
        )

        graph = self.hh_edges.update_graph(graph, attrs_config=None)
        graph["hidden", "to", "hidden"]["edge_length"] = self.edge_length(
            x=(graph["hidden"], graph["hidden"]),
            edge_index=graph["hidden", "to", "hidden"].edge_index,
        )
        graph["hidden", "to", "hidden"]["edge_dirs"] = self.edge_dirs(
            x=(graph["hidden"], graph["hidden"]),
            edge_index=graph["hidden", "to", "hidden"].edge_index,
        )

        graph = self.hd_edges.update_graph(graph, attrs_config=None)
        graph["hidden", "to", "data"]["edge_length"] = self.edge_length(
            x=(graph["hidden"], graph["data"]),
            edge_index=graph["hidden", "to", "data"].edge_index,
        )
        graph["hidden", "to", "data"]["edge_dirs"] = self.edge_dirs(
            x=(graph["hidden"], graph["data"]),
            edge_index=graph["hidden", "to", "data"].edge_index,
        )
        return graph


def build_graph_config(
    dataset_config: dict,
    *,
    resolution: int,
    margin_radius_km: int,
    cutoff_factor: float,
    num_nearest_neighbours: int,
) -> fdl.Config:
    """
    Build a Fiddle config for a limited-area AnemoiGraph.

    Parameters
    ----------
    dataset_config : dict
        Dataset config for AnemoiDatasetNodes.
    resolution : int
        Hidden mesh resolution for LimitedAreaTriNodes.
    margin_radius_km : int
        Margin radius for the limited-area mask.
    cutoff_factor : float
        Cutoff factor for data->hidden edges.
    num_nearest_neighbours : int
        KNN neighbours for hidden->data edges.

    Returns
    -------
    fdl.Config
        Config that builds an AnemoiGraph instance.
    """
    return fdl.Config(
        AnemoiGraph,
        data_nodes=fdl.Config(AnemoiDatasetNodes, name="data", dataset=dataset_config),
        hidden_nodes=fdl.Config(
            LimitedAreaTriNodes,
            name="hidden",
            resolution=resolution,
            reference_node_name="data",
            mask_attr_name="cutout_mask",
            margin_radius_km=margin_radius_km,
        ),
        cutout_mask=fdl.Config(CutOutMask),
        edge_length=fdl.Config(EdgeLength, norm="unit-std"),
        edge_dirs=fdl.Config(EdgeDirection, norm="unit-std"),
        dh_edges=fdl.Config(
            CutOffEdges,
            source_name="data",
            target_name="hidden",
            cutoff_factor=cutoff_factor,
        ),
        hh_edges=fdl.Config(
            MultiScaleEdges,
            source_name="hidden",
            target_name="hidden",
            x_hops=1,
            scale_resolutions=resolution,
        ),
        hd_edges=fdl.Config(
            KNNEdges,
            source_name="hidden",
            target_name="data",
            num_nearest_neighbours=num_nearest_neighbours,
        ),
    )


def test_build_limited_area_anemoi_graph_example():
    """
    Build a minimal Anemoi graph matching WeatherDuck expectations.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    default_dataset_path = (
        "/Users/B280936/Desktop/"
        "cerra-rr-an-oper-0001-mars-5p5km-2017-2017-6h-v3-testing.zarr"
    )
    dataset_path = os.environ.get("ANEMOI_DATASET_PATH", default_dataset_path)
    forcing_path = os.environ.get("ANEMOI_FORCING_DATASET_PATH", dataset_path)
    if not Path(dataset_path).exists() or not Path(forcing_path).exists():
        pytest.skip("Anemoi dataset paths do not exist on disk.")

    dataset_config = {
        "cutout": [
            {"dataset": dataset_path, "thinning": 100},
            {"dataset": forcing_path},
        ],
        "adjust": "all",
        "min_distance_km": 0,
    }

    graph_cache = os.environ.get("WD_CACHE_GRAPH_IN_TESTS")
    graph = _load_or_build_graph(
        graph_cache,
        lambda: fdl.build(
            build_graph_config(
                dataset_config,
                resolution=6,
                margin_radius_km=10,
                cutoff_factor=0.6,
                num_nearest_neighbours=3,
            )
        ).build(),
        label="graph_example",
    )
    assert isinstance(graph, HeteroData)
    assert "data" in graph.node_types
    assert "hidden" in graph.node_types
    assert ("data", "to", "hidden") in graph.edge_types
    assert ("hidden", "to", "hidden") in graph.edge_types
    assert ("hidden", "to", "data") in graph.edge_types


def test_anemoi_datamodule_autoregressive_train_one_epoch():
    """
    Train one epoch with AnemoiNativeGridDataModule and AutoRegressiveForecaster.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    default_dataset_path = (
        "/Users/B280936/Desktop/"
        "cerra-rr-an-oper-0001-mars-5p5km-2017-2017-6h-v3-testing.zarr"
    )
    dataset_path = os.environ.get("ANEMOI_DATASET_PATH", default_dataset_path)
    forcing_path = os.environ.get("ANEMOI_FORCING_DATASET_PATH", dataset_path)
    if not Path(dataset_path).exists() or not Path(forcing_path).exists():
        pytest.skip("Anemoi dataset paths do not exist on disk.")

    dataset_config = {
        "cutout": [
            {"dataset": dataset_path, "thinning": 4},
            {"dataset": forcing_path},
        ],
        "adjust": "all",
        "min_distance_km": 0,
    }
    split_config = {
        "dataset": dataset_config,
        "start": 2017,
        "end": 2017,
        "frequency": "6h",
        "drop": [],
    }

    graph_cache = os.environ.get("WD_CACHE_GRAPH_IN_TESTS")
    graph = _load_or_build_graph(
        graph_cache,
        lambda: fdl.build(
            build_graph_config(
                dataset_config,
                resolution=2,
                margin_radius_km=10,
                cutoff_factor=0.6,
                num_nearest_neighbours=3,
            )
        ).build(),
        label="graph_train",
    )

    start = time.perf_counter()
    dm = AnemoiNativeGridDataModule(
        graph_data=graph,
        training=split_config,
        validation=split_config,
        test=split_config,
        data_frequency="6h",
        data_timestep="6h",
        multistep_input=1,
        rollout=1,
        validation_rollout=1,
        batch_size={"training": 1, "validation": 1, "test": 1},
        num_workers={"training": 0, "validation": 0, "test": 0},
        pin_memory=False,
    )
    dm.setup("fit")
    print(f"[timing] datamodule_setup: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    batch = next(iter(dm.train_dataloader()))
    print(f"[timing] first_batch: {time.perf_counter() - start:.2f}s")

    n_input_data_features = batch["data"].x.shape[1]
    n_hidden_data_features = batch["hidden"].x.shape[1]

    step_model = build_encode_process_decode_model(
        n_input_data_features=n_input_data_features,
        n_output_data_features=n_input_data_features,
        n_hidden_data_features=n_hidden_data_features,
        n_input_trainable_features=0,
        n_hidden_trainable_features=0,
        hidden_dim=64,
    )
    ar_model = AutoRegressiveForecaster(step_predictor=step_model)

    class _AutoRegressiveHarness(pl.LightningModule):
        def __init__(self, model: AutoRegressiveForecaster) -> None:
            super().__init__()
            self.model = model
            self.loss_fn = nn.MSELoss()

        def training_step(self, batch: HeteroData, _batch_idx: int) -> torch.Tensor:
            data_x = batch["data"].x
            ar_graph = batch.clone()
            ar_graph["data"].x_init_states = torch.stack([data_x, data_x], dim=2)
            ar_graph["data"].x_forcing = torch.zeros(
                data_x.shape[0], 0, 1, device=data_x.device
            )
            ar_graph["data"].x_static = torch.zeros(
                data_x.shape[0], 0, device=data_x.device
            )
            ar_graph["data"].y = data_x.unsqueeze(-1)
            y_hat = self.model(ar_graph)
            loss = self.loss_fn(y_hat, ar_graph["data"].y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )
    start = time.perf_counter()
    trainer.fit(_AutoRegressiveHarness(ar_model), datamodule=dm)
    print(f"[timing] trainer_fit: {time.perf_counter() - start:.2f}s")
