import torch

from caveat.encoding.seq_weighting import (
    act_and_dur_inverse_weight,
    act_and_dur_inverse_weights,
    act_inverse_weights,
    seq_inverse_weight,
    seq_max_weight,
    unit_weight,
    unit_weights,
)


def test_unit_weights():
    seq = torch.rand(3, 4, 1)
    weights = unit_weights(seq)
    assert weights.shape == (3, 4)
    assert torch.allclose(weights, torch.ones((3, 4)))


def test_unit_weight():
    seq = torch.rand(3, 4, 1)
    weights = unit_weight(seq)
    assert weights.shape == (3, 1)
    assert torch.allclose(weights, torch.ones((3, 1)))


def test_act_inverse_weights():
    seq = torch.tensor(
        [
            [0, 2, 3, 2, 1, 1],
            [0, 2, 3, 2, 1, 1],
            [0, 2, 2, 1, 1, 1],
            [0, 2, 3, 4, 2, 1],
        ]
    ).unsqueeze(-1)
    weights = act_inverse_weights(seq, trim_eos=False)
    assert weights.shape == (4, 6)
    target = torch.tensor(
        [
            [0.25, 0.125, 1 / 3, 0.125, 0.25, 0.25],
            [0.25, 0.125, 1 / 3, 0.125, 0.25, 0.25],
            [0.25, 0.125, 0.125, 0.25, 0.25, 0.25],
            [0.25, 0.125, 1 / 3, 1, 0.125, 0.25],
        ]
    )
    target = target / target.mean()
    assert torch.allclose(weights, target)


def test_act_inverse_weights_with_trim_eos():
    seq = torch.tensor(
        [
            [0, 2, 3, 2, 1, 1],
            [0, 2, 3, 2, 1, 1],
            [0, 2, 2, 1, 1, 1],
            [0, 2, 3, 4, 2, 1],
        ]
    ).unsqueeze(-1)
    weights = act_inverse_weights(seq, trim_eos=True)
    assert weights.shape == (4, 6)
    target = torch.tensor(
        [
            [0.25, 0.125, 1 / 3, 0.125, 0.25, 0],
            [0.25, 0.125, 1 / 3, 0.125, 0.25, 0],
            [0.25, 0.125, 0.125, 0.25, 0, 0],
            [0.25, 0.125, 1 / 3, 1, 0.125, 0.25],
        ]
    )
    target = target / target.mean()
    assert torch.allclose(weights, target)


def test_seq_inverse_weight():
    seq = torch.tensor(
        [
            [0, 2, 3, 2, 1, 1],
            [0, 2, 3, 2, 1, 1],
            [0, 2, 2, 1, 1, 1],
            [0, 2, 3, 4, 2, 1],
        ]
    ).unsqueeze(-1)
    weights = seq_inverse_weight(seq)
    assert weights.shape == (4, 1)
    target = torch.tensor([[0.5], [0.5], [1], [1]])
    target = target / target.mean()
    assert torch.allclose(weights, target)


def test_seq_max_weight():
    seq = torch.tensor(
        [
            [0, 2, 3, 2, 1, 1],
            [0, 2, 3, 2, 1, 1],
            [0, 2, 2, 1, 1, 1],
            [0, 2, 3, 4, 2, 1],
        ]
    ).unsqueeze(-1)
    weights = seq_max_weight(seq)
    assert weights.shape == (4, 1)
    target = torch.tensor([[1 / 3], [1 / 3], [0.25], [1]])
    target = target / target.mean()
    assert torch.allclose(weights, target)


def test_act_and_dur_inverse_weights():
    acts = torch.tensor([[0, 2, 3, 2, 1, 1], [0, 2, 3, 2, 1, 1]])
    durs = torch.tensor([[0, 0.3, 0.3, 0.4, 0, 0], [0, 0.3, 0.4, 0.3, 0, 0]])
    seq = torch.stack([acts, durs], dim=-1)
    weights = act_and_dur_inverse_weights(seq, trim_eos=False)
    assert weights.shape == (2, 6)
    target = torch.tensor([[2, 3, 1, 1, 2, 2], [2, 3, 1, 3, 2, 2]])
    target = 1 / target
    target = target / target.mean()
    assert torch.allclose(weights, target)


def test_act_and_dur_inverse_weights_with_eos_trim():
    acts = torch.tensor([[0, 2, 3, 2, 1, 1], [0, 2, 3, 2, 1, 1]])
    durs = torch.tensor([[0, 0.3, 0.3, 0.4, 0, 0], [0, 0.3, 0.4, 0.3, 0, 0]])
    seq = torch.stack([acts, durs], dim=-1)
    weights = act_and_dur_inverse_weights(seq, trim_eos=True)
    assert weights.shape == (2, 6)
    target = torch.tensor([[2, 3, 1, 1, 2, 2], [2, 3, 1, 3, 2, 2]])
    target = 1 / target
    target[:, -1] = 0
    target = target / target.mean()
    assert torch.allclose(weights, target)


def test_act_and_dur_inverse_weight():
    acts = torch.tensor(
        [[0, 2, 3, 2, 1, 1], [0, 2, 3, 2, 1, 1], [0, 2, 3, 2, 1, 1]]
    )
    durs = torch.tensor(
        [
            [0, 0.3, 0.3, 0.4, 0, 0],
            [0, 0.3, 0.4, 0.3, 0, 0],
            [0, 0.3, 0.4, 0.3, 0, 0],
        ]
    )
    seq = torch.stack([acts, durs], dim=-1)
    weights = act_and_dur_inverse_weight(seq, trim_eos=False)
    assert weights.shape == (3, 1)
    target = torch.tensor([[1], [2], [2]])
    target = 1 / target
    target = target / target.mean()
    assert torch.allclose(weights, target)
