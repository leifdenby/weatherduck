from __future__ import annotations

from abc import ABC, abstractmethod

from torch_geometric.data import HeteroData

__all__ = ["GraphBuilder"]


class GraphBuilder(ABC):
    """Abstract base class for building WeatherDuck graphs."""

    @abstractmethod
    def __call__(self, coords) -> HeteroData:
        """Build and return a WeatherDuck-compatible graph."""
        raise NotImplementedError
