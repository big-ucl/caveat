import types

import pytest
import torch

from caveat.models.base import Base


def make_stub(free_bits):
    """Minimal stub with only the attribute Base.kld needs."""
    return types.SimpleNamespace(free_bits=free_bits)


def test_kld_zero_free_bits_perfect_posterior():
    # mu=0, log_var=0 → kl_per_dim = -0.5*(1+0-0-1) = 0
    stub = make_stub(free_bits=0.0)
    mu = torch.zeros(2, 4)
    log_var = torch.zeros(2, 4)
    result = Base.kld(stub, mu, log_var)
    assert result.item() == pytest.approx(0.0)


def test_kld_free_bits_clamps_small_kl():
    # mu=0, log_var=0 → raw kl=0 < free_bits=0.5; clamped to 0.5 per dim
    # batch=1, latent_dim=2 → sum=1.0, mean over batch=1 → kld=1.0
    stub = make_stub(free_bits=0.5)
    mu = torch.zeros(1, 2)
    log_var = torch.zeros(1, 2)
    result = Base.kld(stub, mu, log_var)
    assert result.item() == pytest.approx(1.0)


def test_kld_free_bits_no_clamp_when_kl_exceeds_floor():
    # dim 0: mu=2, log_var=0 → kl = -0.5*(1+0-4-1) = 2.0 > 0.1 (not clamped)
    # dim 1: mu=0, log_var=0 → kl = 0.0 < 0.1 (clamped to 0.1)
    # batch=1 → kld = 2.0 + 0.1 = 2.1
    stub = make_stub(free_bits=0.1)
    mu = torch.tensor([[2.0, 0.0]])
    log_var = torch.tensor([[0.0, 0.0]])
    result = Base.kld(stub, mu, log_var)
    assert result.item() == pytest.approx(2.1)
