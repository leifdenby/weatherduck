from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from torch_geometric.data import HeteroData

__all__ = ["GraphProvider"]


class GraphProvider(ABC):
    """Abstract base class for building WeatherDuck graphs."""

    def __init__(self, *, cache: str = "in_memory") -> None:
        """Initialize the graph provider.

        Parameters
        ----------
        cache : str, optional
            Cache mode identifier. The default "in_memory" uses a dict on the
            instance for caching.

        Returns
        -------
        None
        """
        self.cache = cache
        self._cache: dict[str, HeteroData] = {}

    def get_cached(self, graph_id: str) -> HeteroData | None:
        """Return a cached graph for the given id, if present.

        Parameters
        ----------
        graph_id : str
            Graph identifier key for the cache.

        Returns
        -------
        HeteroData | None
            Cached graph when available, otherwise None.
        """
        if self.cache != "in_memory":
            return None
        return self._cache.get(graph_id)

    def set_cached(self, graph_id: str, graph: HeteroData) -> None:
        """Store a graph in the cache.

        Parameters
        ----------
        graph_id : str
            Graph identifier key for the cache.
        graph : HeteroData
            Graph to store in the cache.

        Returns
        -------
        None
        """
        if self.cache != "in_memory":
            return
        self._cache[graph_id] = graph

    @abstractmethod
    def __call__(self, domain_id: str, coords: np.ndarray) -> HeteroData:
        """Build and return a WeatherDuck-compatible graph.

        Parameters
        ----------
        domain_id : str
            Identifier for the data domain. Providers may combine this with their
            parameters to construct a graph id for caching and reuse.
        coords : np.ndarray
            Spatial coordinates of data nodes, shaped [N_data, F_data]. The
            coordinate values are also stored as data-node features (e.g. as
            ``graph["data"].x``) in the returned graph.

        Returns
        -------
        HeteroData
            Graph containing:
            - Node types: 'data', 'hidden'.
            - Node features:
              - graph['data'].x: [N_data, F_data]
              - graph['hidden'].x: [N_hidden, F_hidden]
            - Edge types:
              - ('data','to','hidden'), ('hidden','to','hidden'), ('hidden','to','data')
            - Edge indices:
              - graph[edge_type].edge_index: [2, E]
            - Optional edge attributes:
              - graph[edge_type].edge_attr: [E, D_edge]

            Where:
            - N_data: number of data (grid) nodes (must match coords.shape[0]).
            - N_hidden: number of hidden (mesh) nodes.
            - F_data: data node feature dimension (often 0 at graph construction).
            - F_hidden: hidden node feature dimension (often 0 at graph construction).
            - E: number of edges for the given edge type.
            - D_edge: edge feature dimension (e.g., length and direction features).
        """
        raise NotImplementedError
