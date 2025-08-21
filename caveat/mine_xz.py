import math

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class MutualInformationEstimator(pl.LightningModule):
    def __init__(self, net: nn.Module, **kwargs):
        super().__init__()
        self.net = net
        self.energy_loss = Mine(self.net, alpha=kwargs.get("alpha", 0.01))
        self.lr = kwargs.get("lr", 1e-4)

    def forward(self, y, z):
        return self.energy_loss(y, z)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        y, z = batch
        loss = self.energy_loss(y, z)
        mi = -loss
        self.log_dict({"loss": loss, "mi": mi}, prog_bar=True, logger=True)
        return {"loss": loss, "mi": mi}

    def validation_step(self, batch, batch_idx):
        y, z = batch
        loss = self.energy_loss(y, z)
        mi = -loss
        self.log_dict(
            {"val_loss": loss, "val_mi": mi}, prog_bar=True, logger=True
        )
        return {"val_loss": loss, "val_mi": mi}

    def test_step(self, batch, batch_idx):
        y, z = batch
        loss = self.energy_loss(y, z)
        self.log_dict(
            {"test_loss": loss, "test_mi": -loss}, prog_bar=True, logger=True
        )
        return {"test_loss": loss, "test_mi": -loss}


class Mine(nn.Module):
    def __init__(self, net, alpha=0.01):
        super().__init__()
        self.running_mean = 0
        self.alpha = alpha
        self.net = net

    def forward(self, x, z):
        z_marg = z[torch.randperm(x.shape[0])]

        t = self.net(x, z).mean()
        t_marg = self.net(x, z_marg)

        second_term, self.running_mean = ema_loss(
            t_marg, self.running_mean, self.alpha
        )

        return -t + second_term

    def mi(self, x, z):
        with torch.no_grad():
            mi = -self.forward(x, z)
        return mi


class XZDataset(Dataset):
    def __init__(self, xs: torch.Tensor, zs: torch.Tensor):
        if isinstance(xs, pd.DataFrame):
            xs = xs.values
        if isinstance(zs, pd.DataFrame):
            zs = zs.values
        if isinstance(xs, np.ndarray):
            xs = torch.from_numpy(xs)
        if isinstance(zs, np.ndarray):
            zs = torch.from_numpy(zs).float()
        self.zs = zs
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.zs[idx]


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        val_split: float = 0.1,
        test_split: float = 0.1,
        batch_size: int = 1024,
        num_workers: int = 1,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data = dataset
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mapping = None

    def setup(self, stage) -> None:
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(
            self.data,
            [
                1 - self.val_split - self.test_split,
                self.val_split,
                self.test_split,
            ],
        )
        if self.val_split == 0:
            self.val_dataset = self.train_dataset
        if self.test_split == 0:
            self.test_dataset = self.val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = (
            grad_output
            * input.exp().detach()
            / (running_mean + 1e-6)
            / input.shape[0]
        )
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    return t_log, running_mean
