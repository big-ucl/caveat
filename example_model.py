"""Example model following the harness contract — synthetic binary classification.

Contract functions:
  - get_hyperparameters() -> dict
  - build_model(hp: dict) -> LightningModule
  - get_datamodule(hp: dict) -> LightningDataModule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split


# ---------------------------------------------------------------------------
# Internal model (prefixed with _ to signal it's not part of the contract)
# ---------------------------------------------------------------------------

class _Model(L.LightningModule):
    """Simple MLP for binary classification on synthetic data."""

    def __init__(self, hp: dict):
        super().__init__()
        self.save_hyperparameters(hp)
        hidden_dim = hp.get("hidden_dim", 128)
        dropout = hp.get("dropout", 0.2)
        self.lr = hp.get("lr", 1e-3)

        self.net = nn.Sequential(
            nn.Linear(20, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_accuracy", acc, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class _DM(L.LightningDataModule):
    """DataModule wrapping a synthetic binary classification dataset."""

    def __init__(self, hp: dict):
        super().__init__()
        self.batch_size = hp.get("batch_size", 64)
        self._dataset = None

    def setup(self, stage=None):
        torch.manual_seed(42)
        n, d = 2000, 20
        X = torch.randn(n, d)
        y = ((X[:, 0] + X[:, 1]) > 0).long()
        dataset = TensorDataset(X, y)
        n_val = int(0.2 * n)
        self._train, self._val = random_split(dataset, [n - n_val, n_val])

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=self.batch_size)


# ---------------------------------------------------------------------------
# Contract functions (required by harness)
# ---------------------------------------------------------------------------

def get_hyperparameters() -> dict:
    """Return hyperparameters for this model.

    Returns:
        Dict of hyperparameter name → value.
    """
    return {
        "lr": 1e-3,
        "batch_size": 64,
        "hidden_dim": 128,
        "dropout": 0.2,
        "max_epochs": 20,
    }


def build_model(hp: dict) -> L.LightningModule:
    """Instantiate and return the LightningModule.

    Args:
        hp: Hyperparameters dict from get_hyperparameters().

    Returns:
        An untrained LightningModule.
    """
    return _Model(hp)


def get_datamodule(hp: dict) -> L.LightningDataModule:
    """Instantiate and return the LightningDataModule.

    Args:
        hp: Hyperparameters dict from get_hyperparameters().

    Returns:
        A configured LightningDataModule.
    """
    return _DM(hp)
