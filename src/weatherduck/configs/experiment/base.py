from dataclasses import dataclass

import pytorch_lightning as pl

__all__ = ["Experiment"]


@dataclass
class Experiment:
    pl_module: pl.LightningModule
    data: pl.LightningDataModule
    trainer: pl.Trainer

    def run(self) -> None:
        """Train and evaluate the configured model.

        Returns:
            None.
        """
        self.trainer.fit(self.pl_module, datamodule=self.data)
        self.trainer.test(self.pl_module, datamodule=self.data)
