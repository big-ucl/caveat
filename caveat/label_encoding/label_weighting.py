import torch
from torch import Tensor


def unit_weights(labels: Tensor) -> Tensor:
    return torch.ones_like(labels).float()


def unit_weight(labels: Tensor) -> Tensor:
    return torch.ones((labels.shape[0], 1)).float()


def inverse_weights(labels: Tensor) -> Tensor:
    weights = []
    for i in range(labels.shape[1]):
        _, locs, ws = torch.unique(
            labels[:, i], return_counts=True, return_inverse=True
        )
        weights.append(ws[locs].float())
    weights = torch.stack(weights, dim=1)
    return 1 / weights


def inverse_weight(labels: Tensor) -> Tensor:
    _, locs, ws = torch.unique(
        labels, dim=0, return_counts=True, return_inverse=True
    )
    return ws[locs].float().unsqueeze(-1)


def max_weight(labels: Tensor) -> Tensor:
    weights = inverse_weights(labels)
    return weights.max(dim=-1).values.unsqueeze(-1)
