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


def inverse_first_weight(labels: Tensor) -> Tensor:
    _, locs, ws = torch.unique(
        labels[:, 0], dim=0, return_counts=True, return_inverse=True
    )
    weights = 1 / ws[locs].float()
    weights = weights / weights.mean()
    return weights.unsqueeze(-1)


def effective_num_weight(labels: Tensor, beta: float = 0.999) -> Tensor:
    """
    Effective-number-of-samples reweighting (Cui et al., 2019), which
    softens the extremes of raw inverse-frequency weighting.

    Weight per sample ∝ 1 / E_n, where E_n = (1 - beta^n) / (1 - beta)
    is the "effective number" of samples for that sample's label group,
    n being the raw count of that group.

    As beta -> 1, this approaches raw inverse-frequency weighting.
    As beta -> 0, this approaches uniform weighting (no reweighting).
    Typical values: 0.9, 0.99, 0.999, 0.9999 -- larger for larger/more
    imbalanced datasets.

    Args:
        labels (Tensor): label combinations, [N, label_dim] or [N].
        beta (float): effective-number decay parameter, in [0, 1).

    Returns:
        Tensor: per-sample weights, [N, 1], normalized to mean 1.
    """
    _, locs, ns = torch.unique(
        labels, dim=0, return_counts=True, return_inverse=True
    )
    ns = ns[locs].float()

    effective_num = (1.0 - beta**ns) / (1.0 - beta)
    weights = 1.0 / effective_num
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
