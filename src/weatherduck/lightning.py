import pytorch_lightning as pl
import torch
from torch import nn
from torch_geometric.data import HeteroData

__all__ = ["WeatherDuckModule"]


class WeatherDuckModule(pl.LightningModule):
    """
    Lightning wrapper around a model.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        """Initialize the Lightning wrapper.

        Parameters
        ----------
        model : nn.Module
            Model to train/evaluate.
        lr : float, optional
            Learning rate for the optimizer, by default 1e-3.

        Returns
        -------
        None
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, x: HeteroData) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x : HeteroData
            Input graph batch.

        Returns
        -------
        torch.Tensor
            Model predictions.
        """
        return self.model(x)

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Compute and log training loss.

        Parameters
        ----------
        batch : HeteroData
            Training batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        y = batch["data"].y
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch: HeteroData, batch_idx: int) -> None:
        """Compute and log validation loss.

        Parameters
        ----------
        batch : HeteroData
            Validation batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        None
        """
        y = batch["data"].y
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=1)

    def test_step(self, batch: HeteroData, batch_idx: int) -> None:
        """Compute and log test loss.

        Parameters
        ----------
        batch : HeteroData
            Test batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        None
        """
        y = batch["data"].y
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, prog_bar=True, batch_size=1)

    def configure_optimizers(self):
        """Configure optimizers for training.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
