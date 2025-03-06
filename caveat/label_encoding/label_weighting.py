import torch
from torch import Tensor


def unit_weights(labels: Tensor) -> Tensor:
    weights = torch.ones_like(labels).float()
    return weights


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
    weights = 1 / weights
    weights = weights / weights.mean()
    return weights


def inverse_weight(labels: Tensor) -> Tensor:
    _, locs, ws = torch.unique(
        labels, dim=0, return_counts=True, return_inverse=True
    )
    weights = 1 / ws[locs].float()
    weights = weights / weights.mean()
    return weights.unsqueeze(-1)


def log_inverse_weight(labels: Tensor) -> Tensor:
    _, locs, ws = torch.unique(
        labels, dim=0, return_counts=True, return_inverse=True
    )
    weights = 1 / ws[locs].float()
    weights = torch.log(weights)
    weights = weights / weights.mean()
    return weights.unsqueeze(-1)


def inverse_log_weight(labels: Tensor) -> Tensor:
    _, locs, ws = torch.unique(
        labels, dim=0, return_counts=True, return_inverse=True
    )
    weights = 1 / torch.log(ws[locs] + 0.000001)
    weights = weights / weights.mean()
    return weights.unsqueeze(-1)


def max_weight(labels: Tensor) -> Tensor:
    weights = inverse_weights(labels)
    weights = weights.max(dim=-1).values.unsqueeze(-1)
    weights = weights / weights.mean()
    return weights


def product_weight(labels: Tensor) -> Tensor:
    weights = inverse_weights(labels)
    weights = weights.prod(dim=-1).unsqueeze(-1)
    weights = weights / weights.mean()
    return weights
