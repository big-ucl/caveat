import torch

from caveat.encoding.disc_weighting import (
    act_inverse_weights,
    seq_inverse_weight,
    seq_max_weight,
    unit_weight,
    unit_weights,
)


def test_unit_weights():
    seq = torch.rand(3, 4)
    weights = unit_weights(seq)
    assert weights.shape == (3, 4)
    assert torch.allclose(weights, torch.ones((3, 4)))


def test_unit_weight():
    seq = torch.rand(3, 4)
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
    )
    weights = act_inverse_weights(seq)
    assert weights.shape == (4, 6)
    freqs = torch.tensor(
        [
            [4, 8, 3, 8, 8, 8],
            [4, 8, 3, 8, 8, 8],
            [4, 8, 8, 8, 8, 8],
            [4, 8, 3, 1, 8, 8],
        ]
    )
    target = 1 / freqs
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
    )
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
    )
    weights = seq_max_weight(seq)
    assert weights.shape == (4, 1)
    target = torch.tensor([[1 / 3], [1 / 3], [0.25], [1]])
    target = target / target.mean()
    assert torch.allclose(weights, target)
