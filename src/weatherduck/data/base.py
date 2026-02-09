from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as GeoDataLoader

__all__ = ["BaseWeatherDataModule"]


class BaseWeatherDataModule(pl.LightningDataModule, ABC):
    """Base datamodule that standardizes dataset creation and dataloaders.

    Subclasses should implement ``get_dataset(split)`` and return a
    ``torch.utils.data.Dataset`` that yields ``HeteroData`` samples and
    optionally exposes a ``collate_fn`` attribute for PyG batching.
    """

    def __init__(
        self,
        *,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize the base datamodule.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for loaders, by default 4.
        num_workers : int, optional
            Worker count for loaders, by default 0.
        pin_memory : bool, optional
            Whether to pin memory in loaders, by default False.

        Returns
        -------
        None
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    @abstractmethod
    def get_dataset(self, split: str) -> Dataset:
        """Return the dataset for the given split.

        Parameters
        ----------
        split : str
            Dataset split name ("train", "val", or "test").

        Returns
        -------
        Dataset
            Dataset instance yielding HeteroData samples.
        """
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize datasets for the requested stage.

        Parameters
        ----------
        stage : Optional[str], optional
            Lightning stage hint, by default None.

        Returns
        -------
        None
        """
        if stage in (None, "fit"):
            self.train_ds = self.get_dataset("train")
            self.val_ds = self.get_dataset("val")
        if stage in (None, "test"):
            self.test_ds = self.get_dataset("test")

    def _dataloader(self, ds: Dataset, shuffle: bool) -> GeoDataLoader:
        """Create a dataloader for the provided dataset.

        Parameters
        ----------
        ds : Dataset
            Dataset for the loader.
        shuffle : bool
            Whether to shuffle the dataset.

        Returns
        -------
        GeoDataLoader
            Configured dataloader for HeteroData samples.
        """
        collate_fn = getattr(ds, "collate_fn", None)
        return GeoDataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> GeoDataLoader:
        """Return the training dataloader.

        Returns
        -------
        GeoDataLoader
            Training dataloader.
        """
        return self._dataloader(self.train_ds, shuffle=True)  # type: ignore[arg-type]

    def val_dataloader(self) -> GeoDataLoader:
        """Return the validation dataloader.

        Returns
        -------
        GeoDataLoader
            Validation dataloader.
        """
        return self._dataloader(self.val_ds, shuffle=False)  # type: ignore[arg-type]

    def test_dataloader(self) -> GeoDataLoader:
        """Return the test dataloader.

        Returns
        -------
        GeoDataLoader
            Test dataloader.
        """
        return self._dataloader(self.test_ds, shuffle=False)  # type: ignore[arg-type]
