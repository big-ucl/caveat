import torch
from torch import Tensor


def unit_weights(
    sequences: Tensor, sos_idx: int = 0, eos_idx: int = 1, trim_eos: bool = True
) -> Tensor:
    B, L, A = sequences.shape
    return torch.ones((B, L)).float()


def unit_weight(
    sequences: Tensor, sos_idx: int = 0, eos_idx: int = 1, trim_eos: bool = True
) -> Tensor:
    B, L, A = sequences.shape
    return torch.ones((B, 1)).float()


def act_inverse_weights(
    sequences: Tensor, sos_idx: int = 0, eos_idx: int = 1, trim_eos: bool = True
) -> Tensor:
    activities = sequences[:, :, 0]
    _, locs, ws = torch.unique(
        activities, return_counts=True, return_inverse=True
    )
    # set eos weight to sos weight (ignore trailing eos)
    ws[eos_idx] = ws[sos_idx]

    weights = 1 / ws[locs].float()

    if trim_eos:
        eos_mask = activities == eos_idx
        first_eos = eos_mask.to(torch.long).argmax(dim=-1)
        eos_mask[torch.arange(first_eos.shape[0]), first_eos] = False
        eos_mask = eos_mask.to(torch.float) * -1 + 1  # reverse 1s and 0s
        weights = weights * eos_mask  # apply to weights

    return weights


def seq_inverse_weight(
    sequences: Tensor, sos_idx: int = 0, eos_idx: int = 1, trim_eos: bool = True
) -> Tensor:
    activities = sequences[:, :, 0]
    _, locs, ws = torch.unique(
        activities, dim=0, return_counts=True, return_inverse=True
    )
    return ws[locs].float().unsqueeze(-1)


def seq_max_weight(
    sequences: Tensor, sos_idx: int = 0, eos_idx: int = 1, trim_eos: bool = True
) -> Tensor:
    weights = act_inverse_weights(sequences, sos_idx, eos_idx, trim_eos)
    return weights.max(dim=-1).values.unsqueeze(-1)


act_weight_library = {"unit": unit_weights, "inverse": act_inverse_weights}

seq_weight_library = {
    "unit": unit_weight,
    "inverse": seq_inverse_weight,
    "max": seq_max_weight,
}
