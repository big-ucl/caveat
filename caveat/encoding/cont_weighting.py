import torch
from torch import Tensor

# act level weighting [B, L, 2] -> [B, L]


def unit_weights(
    sequences: Tensor, sos_idx: int = 0, eos_idx: int = 1, trim_eos: bool = True
) -> Tensor:
    B, L, _ = sequences.shape
    weights = torch.ones((B, L)).float()
    if trim_eos:
        activities = sequences[:, :, 0]
        eos_mask = trim_eos_mask(activities, eos_idx)
        weights = weights * eos_mask  # apply to weights
    return weights


def act_inverse_weights(
    sequences: Tensor, sos_idx: int = 0, eos_idx: int = 1, trim_eos: bool = True
) -> Tensor:
    activities = sequences[:, :, 0]
    _, locs, ws = torch.unique(
        activities, return_counts=True, return_inverse=True
    )
    # set sos and eos weights
    ws[sos_idx] = 0
    ws[eos_idx] = 0
    max_act_weight = ws.max()
    ws[eos_idx] = max_act_weight
    ws[sos_idx] = max_act_weight

    weights = 1 / ws[locs].float()

    if trim_eos:
        eos_mask = trim_eos_mask(activities, eos_idx)
        weights = weights * eos_mask  # apply to weights

    weights = weights / weights.mean()

    return weights


def act_and_dur_inverse_weights(
    sequences: Tensor, sos_idx: int = 0, eos_idx: int = 1, trim_eos: bool = True
) -> Tensor:
    activities = sequences[:, :, 0]
    durations = sequences[:, :, 1]
    binned = (durations * 144).to(torch.int)  # 10 minute bins
    combined = torch.stack([activities, binned], dim=-1)
    _, locs, ws = torch.unique(
        combined.view(-1, 2), dim=0, return_counts=True, return_inverse=True
    )
    # set sos and eos weights
    ws[sos_idx] = 0
    ws[eos_idx] = 0
    max_act_weight = ws.max()
    ws[eos_idx] = max_act_weight
    ws[sos_idx] = max_act_weight

    weights = 1 / ws[locs]
    weights = weights.view(sequences.shape[0], -1)

    if trim_eos:
        eos_mask = trim_eos_mask(activities, eos_idx)
        weights = weights * eos_mask  # apply to weights

    weights = weights / weights.mean()
    return weights


# sequence level weighting [B, L, 2] -> [B, 1]


def unit_weight(
    sequences: Tensor, sos_idx: int = 0, eos_idx: int = 1, trim_eos: bool = True
) -> Tensor:
    B, _, _ = sequences.shape
    return torch.ones((B, 1)).float()


def seq_inverse_weight(
    sequences: Tensor, sos_idx: int = 0, eos_idx: int = 1, trim_eos: bool = True
) -> Tensor:
    activities = sequences[:, :, 0]
    _, locs, ws = torch.unique(
        activities, dim=0, return_counts=True, return_inverse=True
    )
    weights = 1 / ws[locs].float()
    weights = weights / weights.mean()
    return weights.unsqueeze(-1)


def act_and_dur_inverse_weight(
    sequences: Tensor, sos_idx: int = 0, eos_idx: int = 1, trim_eos: bool = True
) -> Tensor:
    activities = sequences[:, :, 0]
    durations = sequences[:, :, 1]
    binned = (durations * 144).to(torch.int)  # 10 minute bins
    combined = torch.stack([activities, binned], dim=-1)
    _, locs, ws = torch.unique(
        combined, dim=0, return_counts=True, return_inverse=True
    )
    weights = 1 / ws[locs].float()
    weights = weights / weights.mean()
    return weights.unsqueeze(-1)


def seq_max_weight(
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
        eos_mask = trim_eos_mask(activities, eos_idx)
        weights = weights * eos_mask  # apply to weights

    weights = weights.max(dim=-1).values.unsqueeze(-1)
    weights = weights / weights.mean()
    return weights


# Helper


def trim_eos_mask(activities: Tensor, eos_idx: int = 1) -> Tensor:
    eos_mask = activities == eos_idx
    first_eos = eos_mask.to(torch.long).argmax(dim=-1)
    eos_mask[torch.arange(first_eos.shape[0]), first_eos] = False
    eos_mask = eos_mask.to(torch.float) * -1 + 1  # reverse
    return eos_mask


act_weight_library = {
    "unit": unit_weights,
    "act_inverse": act_inverse_weights,
    "act_dur_inverse": act_and_dur_inverse_weights,
}

seq_weight_library = {
    "unit": unit_weight,
    "act_inverse": seq_inverse_weight,
    "act_dur_inverse": act_and_dur_inverse_weight,
    "max": seq_max_weight,
}
