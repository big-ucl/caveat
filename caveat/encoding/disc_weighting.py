import torch
from torch import Tensor

# act level weighting [B, L] -> [B, L]


def unit_weights(sequences: Tensor) -> Tensor:
    B, L = sequences.shape
    weights = torch.ones((B, L)).float()
    return weights


def act_inverse_weights(sequences: Tensor) -> Tensor:
    activities = sequences
    _, locs, ws = torch.unique(
        activities, return_counts=True, return_inverse=True
    )
    weights = 1 / ws[locs].float()
    weights = weights / weights.mean()

    return weights


# sequence level weighting [B, L] -> [B, 1]


def unit_weight(sequences: Tensor) -> Tensor:
    B, _ = sequences.shape
    return torch.ones((B, 1)).float()


def seq_inverse_weight(sequences: Tensor) -> Tensor:
    activities = sequences
    _, locs, ws = torch.unique(
        activities, dim=0, return_counts=True, return_inverse=True
    )
    weights = 1 / ws[locs].float()
    weights = weights / weights.mean()
    return weights.unsqueeze(-1)


def seq_max_weight(sequences: Tensor) -> Tensor:
    activities = sequences
    _, locs, ws = torch.unique(
        activities, return_counts=True, return_inverse=True
    )
    # set eos weight to sos weight (ignore trailing eos)
    weights = 1 / ws[locs].float()

    weights = weights.max(dim=-1).values.unsqueeze(-1)
    weights = weights / weights.mean()
    return weights


act_weight_library = {"unit": unit_weights, "act_inverse": act_inverse_weights}

seq_weight_library = {
    "unit": unit_weight,
    "act_inverse": seq_inverse_weight,
    "max": seq_max_weight,
}
