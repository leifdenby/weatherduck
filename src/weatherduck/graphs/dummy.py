from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .base import GraphProvider

__all__ = ["build_dummy_weather_graph", "DummyGraphProvider"]


def build_dummy_weather_graph(
    data_coords: np.ndarray,
    num_hidden_nodes: int = 32,
    edge_attr_dim: int = 2,
) -> HeteroData:
    """Build a minimal heterogeneous graph with random connectivity.

    Parameters
    ----------
    data_coords : np.ndarray
        Data-node coordinates with shape ``[N_data, F_data]``; these values are
        written to ``graph["data"].x``.
    num_hidden_nodes : int, optional
        Number of hidden nodes to create.
    edge_attr_dim : int, optional
        Edge attribute dimension for all edge types.
    Returns
    -------
    HeteroData
        Graph with node types ``data`` and ``hidden`` and three edge relations
        (``data→hidden``, ``hidden→hidden``, ``hidden→data``). Data node features
        are set from ``data_coords``. Hidden node features are sampled uniformly
        within the per-dimension min/max range of data features. Edge indices
        are generated with dense random fanout and edge attributes are sampled
        from a standard normal distribution.
    """
    graph = HeteroData()
    if data_coords.ndim != 2:
        raise ValueError("data_coords must be a 2D array.")
    num_data_nodes = data_coords.shape[0]
    graph["data"].x = torch.as_tensor(data_coords, dtype=torch.float32)
    data_min = torch.min(graph["data"].x, dim=0).values
    data_max = torch.max(graph["data"].x, dim=0).values
    if data_min.numel() == 0:
        graph["hidden"].x = torch.zeros(num_hidden_nodes, 0)
    else:
        hidden_x = (
            torch.rand(num_hidden_nodes, data_min.shape[0]) * (data_max - data_min)
            + data_min
        )
        graph["hidden"].x = hidden_x

    def dense_edges(n_src: int, n_dst: int, fanout: int) -> torch.Tensor:
        """Generate random dense edge indices.

        Parameters
        ----------
        n_src : int
            Number of source nodes.
        n_dst : int
            Number of destination nodes.
        fanout : int
            Number of edges per source node.

        Returns
        -------
        torch.Tensor
            Edge indices of shape [2, n_edges].
        """
        src = torch.arange(n_src).repeat_interleave(fanout)
        dst_choices = torch.randint(0, n_dst, (n_src * fanout,))
        return torch.stack([src, dst_choices], dim=0)

    graph["data", "to", "hidden"].edge_index = dense_edges(
        num_data_nodes, num_hidden_nodes, fanout=4
    )
    graph["hidden", "to", "hidden"].edge_index = dense_edges(
        num_hidden_nodes, num_hidden_nodes, fanout=6
    )
    graph["hidden", "to", "data"].edge_index = dense_edges(
        num_hidden_nodes, num_data_nodes, fanout=4
    )

    for key in [
        ("data", "to", "hidden"),
        ("hidden", "to", "hidden"),
        ("hidden", "to", "data"),
    ]:
        num_edges = graph[key].edge_index.shape[1]
        graph[key].edge_attr = torch.randn(num_edges, edge_attr_dim)

    return graph


class DummyGraphProvider(GraphProvider):
    """Builds small synthetic graphs for quick iteration and testing."""

    def __init__(
        self,
        *,
        num_data_nodes: int | None = None,
        num_hidden_nodes: int | None = None,
        edge_attr_dim: int = 2,
        cache: str = "in_memory",
    ) -> None:
        """Initialize the dummy graph provider.

        Parameters
        ----------
        num_data_nodes : int | None, optional
            Number of data nodes; inferred from coords if None.
        num_hidden_nodes : int | None, optional
            Number of hidden nodes; defaults to half of data nodes.
        edge_attr_dim : int, optional
            Edge attribute dimension.
        cache : str, optional
            Cache mode identifier, by default "in_memory".

        Returns
        -------
        None
        """
        super().__init__(cache=cache)
        self.num_data_nodes = num_data_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.edge_attr_dim = edge_attr_dim

    def __call__(self, domain_id: str, coords: np.ndarray) -> HeteroData:
        """Build a dummy graph for given coords.

        Parameters
        ----------
        domain_id : str
            Identifier for the data domain.
        coords : np.ndarray
            Coordinates/features array shaped [N_data, F_data].

        Returns
        -------
        HeteroData
            Dummy heterogenous graph with:
            - Node types:
              - `data`: `graph["data"].x` with shape `[N_data, F_data]`, values
                taken directly from `coords` (coordinates become features).
              - `hidden`: `graph["hidden"].x` with shape `[N_hidden, F_hidden]`,
                values sampled from a standard normal distribution.
            - Edge types (dense random fanout):
              - `("data","to","hidden")` with `edge_index` `[2, E_dh]` and
                `edge_attr` `[E_dh, edge_attr_dim]` sampled from a standard normal.
              - `("hidden","to","hidden")` with `edge_index` `[2, E_hh]` and
                `edge_attr` `[E_hh, edge_attr_dim]` sampled from a standard normal.
              - `("hidden","to","data")` with `edge_index` `[2, E_hd]` and
                `edge_attr` `[E_hd, edge_attr_dim]` sampled from a standard normal.
        """
        graph_id = f"dummy__{domain_id}"
        cached = self.get_cached(graph_id)
        if cached is not None:
            return cached.clone()
        if coords.ndim != 2:
            raise ValueError("coords must be a 2D array.")
        num_data_nodes = (
            coords.shape[0] if self.num_data_nodes is None else self.num_data_nodes
        )
        num_hidden_nodes = (
            max(1, num_data_nodes // 2)
            if self.num_hidden_nodes is None
            else self.num_hidden_nodes
        )
        graph = build_dummy_weather_graph(
            data_coords=coords,
            num_hidden_nodes=num_hidden_nodes,
            edge_attr_dim=self.edge_attr_dim,
        )
        graph.graph_id_str = graph_id
        self.set_cached(graph_id, graph)
        return graph

    def node_static_feature_dim(self, node_type: str) -> int:
        """Return the static feature dimension for a node type.

        Parameters
        ----------
        node_type : str
            Node type name.

        Returns
        -------
        int
            Static feature dimension for the node type.
        """
        if node_type == "data":
            return 2
        if node_type == "hidden":
            return 2
        raise ValueError(f"Unsupported node type: {node_type}")

    def edge_static_feature_dim(self, edge_type: tuple[str, str, str]) -> int:
        """Return the static feature dimension for an edge type.

        Parameters
        ----------
        edge_type : tuple[str, str, str]
            Edge type tuple.

        Returns
        -------
        int
            Static feature dimension for the edge type.
        """
        if edge_type in {
            ("data", "to", "hidden"),
            ("hidden", "to", "hidden"),
            ("hidden", "to", "data"),
        }:
            return self.edge_attr_dim
        raise ValueError(f"Unsupported edge type: {edge_type}")
