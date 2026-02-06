from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from torch_geometric.data import HeteroData

__all__ = ["GraphBuilder"]


class GraphBuilder(ABC):
    """Abstract base class for building WeatherDuck graphs."""

    @abstractmethod
    def __call__(self, coords: np.ndarray) -> HeteroData:
        """Build and return a WeatherDuck-compatible graph.

        Parameters
        ----------
        coords : np.ndarray
            Spatial coordinates for data nodes, shaped [N_data, 2]. Implementations
            may use this to infer node counts or construct geometry-aware edges.

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
