import torch

from caveat.label_encoding.label_weighting import (
    inverse_log_weight,
    inverse_weight,
    inverse_weights,
    log_inverse_weight,
    max_weight,
    product_weight,
    unit_weight,
    unit_weights,
)


def test_unit_weights():
    labels = torch.rand(3, 2)
    weights = unit_weights(labels)
    assert weights.shape == (3, 2)
    assert torch.allclose(weights, torch.ones_like(labels))


def test_unit_weight():
    labels = torch.rand(3, 2)
    weights = unit_weight(labels)
    assert weights.shape == (3, 1)
    assert torch.allclose(weights, torch.ones((3, 1)))


def test_inverse_weights():
    labels = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = inverse_weights(labels)
    assert weights.shape == (3, 2)
    target = torch.tensor([[0.5, 0.5], [1, 1], [0.5, 0.5]])
    target = target / target.mean()
    assert torch.allclose(weights, target)


def test_inverse_weight():
    labels = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = inverse_weight(labels)
    assert weights.shape == (3, 1)
    target = torch.tensor([[0.5], [1], [0.5]])
    target = target / target.mean()
    assert torch.allclose(weights, target)


def test_max_weight():
    labels = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = max_weight(labels)
    assert weights.shape == (3, 1)
    target = torch.tensor([[0.5], [1], [0.5]])
    target = target / target.mean()
    assert torch.allclose(weights, target)


def test_log_inverse_weight():
    labels = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = log_inverse_weight(labels)
    assert weights.shape == (3, 1)
    target = torch.log(torch.tensor([[0.5], [1], [0.5]]))
    target = target / target.mean()
    assert torch.allclose(weights, target)


def test_inverse_log_weight():
    labels = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = inverse_log_weight(labels)
    assert weights.shape == (3, 1)
    target = torch.tensor([[0.5], [1], [0.5]])
    target = 1 / torch.log((1 / target) + 0.000001)
    target = target / target.mean()
    assert torch.allclose(weights, target, atol=1e-6)


def test_product_inverse_weight():
    labels = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = product_weight(labels)
    assert weights.shape == (3, 1)
    target = torch.tensor([[0.25], [1], [0.25]])
    target = target / target.mean()
    assert torch.allclose(weights, target)
