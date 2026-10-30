from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.distributions import OneHotCategorical
from torch.utils.data import DataLoader, Dataset


class BatchLoader:
    """Iterates pre-built tensors by index. No workers, no collate, no transfers."""

    def __init__(self, dataset, indices, batch_size, shuffle, drop_last=False):
        self.dataset = dataset
        self.indices = indices  # LongTensor on the same device as the data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.indices)
        return (
            n // self.batch_size if self.drop_last else -(-n // self.batch_size)
        )

    def __iter__(self):
        idx = self.indices
        if self.shuffle:
            idx = idx[torch.randperm(len(idx), device=idx.device)]
        for i in range(len(self)):
            yield self.dataset.get_batch(
                idx[i * self.batch_size : (i + 1) * self.batch_size]
            )


class DataModule(LightningDataModule):
    def __init__(
        self,
        data: Dataset,
        val_split: float = 0.1,
        test_split: Optional[float] = None,
        train_batch_size: int = 1024,
        val_batch_size: int = 1024,
        test_batch_size: int = 1024,
        device: str = "cuda",
        drop_last: bool = True,
        seed: int = 0,
        **kwargs,  # swallows num_workers / pin_memory from existing configs
    ):
        super().__init__()
        self.data = data
        self.val_split = val_split
        self.test_split = test_split
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.device = device
        self.drop_last = drop_last
        self.mapping = None
        self.seed = seed

    def setup(self, stage: Optional[str] = None) -> None:
        self.data.to(self.device)  # move every tensor once
        n = len(self.data)
        perm = torch.randperm(
            n, generator=torch.Generator().manual_seed(self.seed)
        ).to(self.device)
        n_val = int(n * self.val_split)
        n_test = int(n * self.test_split) if self.test_split else 0
        self.val_idx = perm[:n_val]
        self.test_idx = perm[n_val : n_val + n_test] if n_test else self.val_idx
        self.train_idx = perm[n_val + n_test :]

    def train_dataloader(self):
        return BatchLoader(
            self.data,
            self.train_idx,
            self.train_batch_size,
            shuffle=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return BatchLoader(
            self.data, self.val_idx, self.val_batch_size, shuffle=False
        )

    def test_dataloader(self):
        return BatchLoader(
            self.data, self.test_idx, self.test_batch_size, shuffle=False
        )


class ZNormDataset(Dataset):
    def __init__(self, num_samples: int, latent_dim: int):
        self.z = torch.randn(num_samples, latent_dim)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], torch.tensor(0)


class ZCatDataset(Dataset):
    def __init__(
        self, num_samples: int, latent_dim: int, latent_categories: int
    ):
        uniform = torch.ones((num_samples, latent_dim, latent_categories))
        uniform /= uniform.sum(dim=-1, keepdim=True)
        self.z = OneHotCategorical(uniform).sample()

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], torch.tensor(0)


def build_latent_dataloader(
    num_samples,
    latent_dim: int,
    latent_categories: Optional[int] = None,
    batch_size: int = 256,
    num_workers: int = 4,
):
    if latent_categories is None:
        return DataLoader(
            ZNormDataset(num_samples, latent_dim),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
        )
    else:
        return DataLoader(
            ZCatDataset(num_samples, latent_dim, latent_categories),
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


def build_latent_conditional_dataloader(
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


class CustomGenDataset(Dataset):
    def __init__(self, attributes: torch.Tensor, z: torch.Tensor):
        self.z = z
        self.attributes = attributes

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        return self.z[idx], self.attributes[idx]


def build_custom_gen_dataloader(
    attributes: torch.Tensor,
    z: torch.Tensor,
    batch_size: int = 256,
    num_workers: int = 4,
):
    print(
        f"Building dataloader with {len(attributes)} samples, latent_dim={len(z)}, batch_size={batch_size}, num_workers={num_workers}"
    )
    return DataLoader(
        CustomGenDataset(attributes, z),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
