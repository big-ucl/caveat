from typing import Optional, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DataModule(LightningDataModule):
    def __init__(
        self,
        data: Dataset,
        val_split: float = 0.1,
        test_split: Optional[float] = None,
        train_batch_size: int = 128,
        val_batch_size: int = 128,
        test_batch_size: int = 128,
        num_workers: int = 1,
        pin_memory: bool = False,
        **kwargs,
    ):
        """Torch DataModule.

        Args:
            data (Dataset): Data
            val_split (float, optional): _description_. Defaults to None.
            test_split (Optional[float], optional): _description_. Defaults to 0.1.
            train_batch_size (int, optional): _description_. Defaults to 128.
            val_batch_size (int, optional): _description_. Defaults to 128.
            test_batch_size (int, optional): _description_. Defaults to 128.
            num_workers (int, optional): _description_. Defaults to 0.
            pin_memory (bool, optional): _description_. Defaults to False.
        """
        super().__init__()

        self.data = data
        self.val_split = val_split
        self.test_split = test_split
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mapping = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.test_split is None:
            (self.train_dataset, self.val_dataset) = (
                torch.utils.data.random_split(
                    self.data, [1 - self.val_split, self.val_split]
                )
            )
            self.test_dataset = self.val_dataset
        else:
            (self.train_dataset, self.val_dataset, self.test_dataset) = (
                torch.utils.data.random_split(
                    self.data,
                    [
                        1 - self.val_split - self.test_split,
                        self.val_split,
                        self.test_split,
                    ],
                )
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def test_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )


class ZDataset(Dataset):
    def __init__(self, num_samples: int, latent_dim: int):
        self.z = torch.randn(num_samples, latent_dim)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], torch.tensor(0)


def build_predict_dataloader(
    num_samples, latent_dim: int, batch_size: int = 256, num_workers: int = 4
):
    return DataLoader(
        ZDataset(num_samples, latent_dim),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )


class ConditionalDataset(Dataset):
    def __init__(self, attributes: torch.Tensor, latent_dim: int):
        self.z = torch.randn(len(attributes), latent_dim)
        self.attributes = attributes

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        return self.z[idx], self.attributes[idx]


def build_conditional_dataloader(
    attributes: torch.Tensor,
    latent_dim: int,
    batch_size: int = 256,
    num_workers: int = 4,
):
    print(
        f"Building dataloader with {len(attributes)} samples, latent_dim={latent_dim}, batch_size={batch_size}, num_workers={num_workers}"
    )
    return DataLoader(
        ConditionalDataset(attributes, latent_dim),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
