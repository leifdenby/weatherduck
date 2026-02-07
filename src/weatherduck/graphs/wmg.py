from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch_geometric.data import HeteroData
from weather_model_graphs.create.archetype import (
    create_graphcast_graph,
    create_keisler_graph,
    create_oskarsson_hierarchical_graph,
)

from .base import GraphProvider

__all__ = ["WMGGraphProvider"]


class WMGGraphProvider(GraphProvider):
    """Provide WeatherDuck graphs using weather-model-graphs utilities."""

    def __init__(
        self,
        *,
        kind: Literal["keisler", "graphcast", "oskarsson_hierarchical"] = "graphcast",
        mesh_node_distance: float,
        level_refinement_factor: int = 3,
        max_num_levels: int | None = None,
        coords_crs=None,
        graph_crs=None,
    ) -> None:
        """Initialize the WMG graph provider.

        Parameters
        ----------
        kind : Literal["keisler", "graphcast", "oskarsson_hierarchical"], optional
            Graph archetype to build.
        mesh_node_distance : float, optional
            Base mesh node spacing in coordinate units.
        level_refinement_factor : int, optional
            Refinement factor between mesh levels.
        max_num_levels : int | None, optional
            Maximum number of mesh levels.
        coords_crs : Any, optional
            Coordinate CRS of input coords.
        graph_crs : Any, optional
            CRS to use for graph construction.

        Returns
        -------
        None
        """
        self.kind = kind
        self.mesh_node_distance = mesh_node_distance
        self.level_refinement_factor = level_refinement_factor
        self.max_num_levels = max_num_levels
        self.coords_crs = coords_crs
        self.graph_crs = graph_crs

    def __call__(self, coords: np.ndarray) -> HeteroData:
        """Build a graph from spatial coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Array of shape [N, 2] with spatial coordinates.

        Returns
        -------
        HeteroData
            WeatherDuck-compatible graph.
        """
        nx_graph = self._build_networkx_graph(coords)
        return _to_heterodata(nx_graph)

    def _build_networkx_graph(self, coords: np.ndarray):
        """Build a weather-model-graphs networkx graph.

        Parameters
        ----------
        coords : np.ndarray
            Array of shape [N, 2] with spatial coordinates.

        Returns
        -------
        networkx.DiGraph
            Constructed networkx graph.
        """

        if self.kind == "keisler":
            return create_keisler_graph(
                coords=coords,
                mesh_node_distance=self.mesh_node_distance,
                coords_crs=self.coords_crs,
                graph_crs=self.graph_crs,
            )
        if self.kind == "oskarsson_hierarchical":
            return create_oskarsson_hierarchical_graph(
                coords=coords,
                mesh_node_distance=self.mesh_node_distance,
                level_refinement_factor=self.level_refinement_factor,
                max_num_levels=self.max_num_levels,
                coords_crs=self.coords_crs,
                graph_crs=self.graph_crs,
            )
        return create_graphcast_graph(
            coords=coords,
            mesh_node_distance=self.mesh_node_distance,
            level_refinement_factor=self.level_refinement_factor,
            max_num_levels=self.max_num_levels,
            coords_crs=self.coords_crs,
            graph_crs=self.graph_crs,
        )


def _to_heterodata(nx_graph) -> HeteroData:
    """Convert a weather-model-graphs networkx graph to HeteroData.

    Parameters
    ----------
    nx_graph : networkx.DiGraph
        Input graph with node/edge attributes.

    Returns
    -------
    HeteroData
        Converted heterogeneous graph.
    """
    node_types = {}
    data_nodes = []
    hidden_nodes = []
    data_pos = []
    hidden_pos = []

    for node_id, attrs in nx_graph.nodes(data=True):
        node_type = attrs.get("type", "grid")
        if node_type == "mesh":
            idx = len(hidden_nodes)
            hidden_nodes.append(node_id)
            hidden_pos.append(np.asarray(attrs.get("pos", [0.0, 0.0]), dtype=float))
            node_types[node_id] = ("hidden", idx)
        else:
            idx = len(data_nodes)
            data_nodes.append(node_id)
            data_pos.append(np.asarray(attrs.get("pos", [0.0, 0.0]), dtype=float))
            node_types[node_id] = ("data", idx)

    graph = HeteroData()
    graph["data"].x = (
        torch.from_numpy(np.stack(data_pos, axis=0)).to(torch.float32)
        if data_pos
        else torch.zeros(0, 2)
    )
    graph["hidden"].x = (
        torch.from_numpy(np.stack(hidden_pos, axis=0)).to(torch.float32)
        if hidden_pos
        else torch.zeros(0, 2)
    )

    edge_buckets: dict[tuple[str, str, str], list[tuple[int, int, np.ndarray]]] = {
        ("data", "to", "hidden"): [],
        ("hidden", "to", "hidden"): [],
        ("hidden", "to", "data"): [],
    }

    for u, v, attrs in nx_graph.edges(data=True):
        src_type, src_idx = node_types[u]
        dst_type, dst_idx = node_types[v]
        component = attrs.get("component")
        if component == "g2m":
            key = ("data", "to", "hidden")
        elif component == "m2g":
            key = ("hidden", "to", "data")
        elif component == "m2m":
            key = ("hidden", "to", "hidden")
        else:
            if src_type == "data" and dst_type == "hidden":
                key = ("data", "to", "hidden")
            elif src_type == "hidden" and dst_type == "data":
                key = ("hidden", "to", "data")
            elif src_type == "hidden" and dst_type == "hidden":
                key = ("hidden", "to", "hidden")
            else:
                continue
        edge_len = float(attrs.get("len", 0.0))
        vdiff = np.asarray(attrs.get("vdiff", [0.0, 0.0]), dtype=float)
        edge_feat = np.concatenate([np.atleast_1d(edge_len), vdiff], axis=0)
        edge_buckets[key].append((src_idx, dst_idx, edge_feat))

    for key, edges in edge_buckets.items():
        if not edges:
            graph[key].edge_index = torch.zeros(2, 0, dtype=torch.long)
            graph[key].edge_attr = torch.zeros(0, 3)
            continue
        src, dst, feats = zip(*edges)
        graph[key].edge_index = torch.tensor([src, dst], dtype=torch.long)
        graph[key].edge_attr = torch.from_numpy(np.stack(feats, axis=0)).to(
            torch.float32
        )

    return graph
