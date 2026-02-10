import pytest
import torch

from weatherduck.data.dummy import DummyWeatherDataset
from weatherduck.graphs import DummyGraphProvider, WMGGraphProvider


def test_dummy_graph_provider_from_dataset():
    """Build a dummy graph from dataset-provided coordinates.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    num_data_nodes = 16
    ds = DummyWeatherDataset(
        num_samples=1,
        num_data_nodes=num_data_nodes,
        n_input_data_features=4,
        n_output_data_features=2,
        graph_provider=DummyGraphProvider(),
    )
    graph = ds[0]
    assert graph["data"].num_nodes == num_data_nodes
    assert ("data", "to", "hidden") in graph.edge_types
    assert ("hidden", "to", "hidden") in graph.edge_types
    assert ("hidden", "to", "data") in graph.edge_types
    assert graph["data"].x.shape[0] == num_data_nodes


def test_wmg_graph_provider_from_dataset():
    """Build a weather-model-graphs-based graph from dataset coordinates.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pytest.importorskip("weather_model_graphs")

    num_data_nodes = 16
    ds = DummyWeatherDataset(
        num_samples=1,
        num_data_nodes=num_data_nodes,
        n_input_data_features=4,
        n_output_data_features=2,
        graph_provider=WMGGraphProvider(mesh_node_distance=1.0),
    )
    graph = ds[0]
    assert graph["data"].num_nodes == num_data_nodes
    assert ("data", "to", "hidden") in graph.edge_types
    assert ("hidden", "to", "hidden") in graph.edge_types
    assert ("hidden", "to", "data") in graph.edge_types
    assert graph["data"].x.shape[0] == num_data_nodes
    assert torch.isfinite(graph["data"].x).all()
