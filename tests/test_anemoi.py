import os
import time
from pathlib import Path

import fiddle as fdl
import pytest
import torch
from anemoi.datasets import open_dataset
from loguru import logger
from torch_geometric.data import HeteroData

from weatherduck.configs.experiment.anemoi import (
    create_lam_graph_builder,
    experiment_factory,
)


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
    graph_builder = fdl.build(
        create_lam_graph_builder(
            dataset_config,
            resolution=6,
            margin_radius_km=10,
            cutoff_factor=0.6,
            num_nearest_neighbours=3,
        )
    )
    graph = _load_or_build_graph(
        graph_cache,
        graph_builder.create,
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
    dataset = open_dataset(dataset_config)
    n_input_data_features = len(dataset.name_to_index)

    experiment = fdl.build(
        experiment_factory(
            dataset_config=dataset_config,
            split_config=split_config,
            n_input_data_features=n_input_data_features,
            resolution=2,
            margin_radius_km=10,
            cutoff_factor=0.6,
            num_nearest_neighbours=3,
        )
    )
    start = time.perf_counter()
    experiment.run()
    print(f"[timing] trainer_fit: {time.perf_counter() - start:.2f}s")
