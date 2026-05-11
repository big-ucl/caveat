"""Integration tests for CollapseMonitor using a real CVAEContLSTM model.

These tests complement the mock-based tests in test_callbacks.py by exercising
the actual encode/predict paths of the model, catching interface mismatches that
mocks cannot detect.
"""
import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from caveat.callbacks import CollapseMonitor
from caveat.models.continuous.cvae_lstm import CVAEContLSTM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LENGTH = 8
N_ACTIVITIES = 5
LATENT_DIM = 4
HIDDEN_SIZE = 16
HIDDEN_N = 1
N_LABEL_COLS = 2
LABEL_EMBED_SIZES = [3, 4]  # 3 categories for label0, 4 for label1


@pytest.fixture
def model():
    """Minimal CVAEContLSTM with small hidden size for fast tests."""
    return CVAEContLSTM(
        in_shape=(LENGTH, N_ACTIVITIES),
        encodings=N_ACTIVITIES,
        labels_size=N_LABEL_COLS,
        label_embed_sizes=LABEL_EMBED_SIZES,
        sos=0,
        latent_dim=LATENT_DIM,
        hidden_size=HIDDEN_SIZE,
        hidden_n=HIDDEN_N,
        dropout=0.0,
    )


def make_batch(n=4, varied_conditions=True):
    """Build a (x, c) batch compatible with CVAEContLSTM.

    x: [N, LENGTH, 2]  — activity index (int as float) + duration
    c: [N, N_LABEL_COLS] long — label indices within embed sizes
    """
    x = torch.zeros(n, LENGTH, 2)
    x[:, :, 0] = torch.randint(0, N_ACTIVITIES, (n, LENGTH)).float()
    x[:, :, 1] = torch.rand(n, LENGTH)

    if varied_conditions:
        c = torch.stack(
            [
                torch.randint(0, LABEL_EMBED_SIZES[0], (n,)),
                torch.randint(0, LABEL_EMBED_SIZES[1], (n,)),
            ],
            dim=1,
        )
        # Guarantee at least two distinct rows so the derangement check passes.
        c[0] = torch.tensor([0, 0])
        c[1] = torch.tensor([1, 1])
    else:
        c = torch.zeros(n, N_LABEL_COLS, dtype=torch.long)

    return x, c


def make_trainer(current_epoch=0):
    trainer = MagicMock()
    trainer.current_epoch = current_epoch
    return trainer


# ---------------------------------------------------------------------------
# on_validation_batch_end with real model
# ---------------------------------------------------------------------------


def test_batch_end_accumulates_mu_log_var(model):
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    trainer = make_trainer()
    batch = make_batch()

    cb.on_validation_batch_end(trainer, model, None, batch, 0)

    assert len(cb._mus) == 1
    assert len(cb._log_vars) == 1
    assert cb._mus[0].shape == (4, LATENT_DIM)
    assert cb._log_vars[0].shape == (4, LATENT_DIM)


def test_batch_end_accumulates_across_multiple_batches(model):
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    trainer = make_trainer()

    for i in range(3):
        cb.on_validation_batch_end(trainer, model, None, make_batch(), i)

    assert len(cb._mus) == 3
    assert len(cb._kl_per_dim) == 3


def test_batch_end_accumulates_decoder_sensitivity_buffers(model):
    """With varied conditions the decoder swap buffers should be populated."""
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    trainer = make_trainer()
    cb.on_validation_batch_end(
        trainer, model, None, make_batch(varied_conditions=True), 0
    )

    assert len(cb._decoder_swap_mse) == 1
    assert len(cb._decoder_out_var) == 1
    assert math.isfinite(cb._decoder_swap_mse[0])
    assert math.isfinite(cb._decoder_out_var[0])


def test_batch_end_skips_sensitivity_when_single_condition(model, capsys):
    """All-same conditions → buffers stay empty and a warning is printed."""
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    trainer = make_trainer()
    cb.on_validation_batch_end(
        trainer, model, None, make_batch(varied_conditions=False), 0
    )

    assert cb._decoder_swap_mse == []
    assert cb._decoder_out_var == []
    out = capsys.readouterr().out
    assert "CollapseMonitor" in out
    assert "skipping decoder sensitivity" in out


def test_batch_end_raises_on_mu_shape_mismatch(model):
    """Batch size mismatch between mu and c raises ValueError."""
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    trainer = make_trainer()
    x, c = make_batch(n=4, varied_conditions=True)
    # Manually corrupt: override encode to return mu with wrong batch size
    import unittest.mock as mock

    bad_mu = torch.randn(3, LATENT_DIM)  # 3 rows, but c has 4
    bad_log_var = torch.zeros(3, LATENT_DIM)
    with mock.patch.object(model, "encode", return_value=(bad_mu, bad_log_var)):
        with pytest.raises(ValueError, match="Batch size mismatch"):
            cb.on_validation_batch_end(trainer, model, None, (x, c), 0)


# ---------------------------------------------------------------------------
# on_validation_epoch_end with real model
# ---------------------------------------------------------------------------

EXPECTED_METRIC_KEYS = {
    "collapse/active_units_pct",
    "collapse/n_active_dims",
    "collapse/kl_collapsed_dims",
    "collapse/kl_mean",
    "collapse/kl_min",
    "collapse/decoder_sensitivity",
    "collapse/mean_posterior_var",
}


def _run_epoch(model, cb, current_epoch=0, n_batches=2, varied_conditions=True):
    """Populate buffers via batch_end and call epoch_end, returning logged metrics."""
    trainer = make_trainer(current_epoch=current_epoch)
    for i in range(n_batches):
        cb.on_validation_batch_end(
            trainer,
            model,
            None,
            make_batch(varied_conditions=varied_conditions),
            i,
        )
    with patch.object(model, "log_dict") as mock_log:
        cb.on_validation_epoch_end(trainer, model)
    return mock_log


def test_epoch_end_logs_all_keys(model):
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    mock_log = _run_epoch(model, cb)

    mock_log.assert_called_once()
    logged = mock_log.call_args[0][0]
    assert set(logged.keys()) == EXPECTED_METRIC_KEYS


def test_epoch_end_values_are_finite(model):
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    mock_log = _run_epoch(model, cb)

    logged = mock_log.call_args[0][0]
    for key, val in logged.items():
        if key != "collapse/decoder_sensitivity":
            assert math.isfinite(val), f"{key} = {val} is not finite"


def test_epoch_end_decoder_sensitivity_finite_with_varied_conditions(model):
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    mock_log = _run_epoch(model, cb, varied_conditions=True)

    logged = mock_log.call_args[0][0]
    assert math.isfinite(logged["collapse/decoder_sensitivity"])


def test_epoch_end_decoder_sensitivity_nan_when_no_varied_conditions(model):
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    mock_log = _run_epoch(model, cb, varied_conditions=False)

    logged = mock_log.call_args[0][0]
    assert math.isnan(logged["collapse/decoder_sensitivity"])


def test_epoch_end_resets_buffers(model):
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    _run_epoch(model, cb)

    assert cb._mus == []
    assert cb._log_vars == []
    assert cb._conditions == []
    assert cb._kl_per_dim == []
    assert cb._decoder_swap_mse == []
    assert cb._decoder_out_var == []


def test_epoch_end_skips_non_check_epoch(model):
    cb = CollapseMonitor({"check_every_n_epochs": 5})
    trainer = make_trainer(current_epoch=3)

    cb.on_validation_batch_end(trainer, model, None, make_batch(), 0)
    with patch.object(model, "log_dict") as mock_log:
        cb.on_validation_epoch_end(trainer, model)

    mock_log.assert_not_called()


# ---------------------------------------------------------------------------
# Full pipeline: multiple check epochs
# ---------------------------------------------------------------------------


def test_full_pipeline_two_check_epochs(model):
    """Run two consecutive check epochs and verify metrics logged each time."""
    cb = CollapseMonitor({"check_every_n_epochs": 5})
    log_call_count = 0

    for epoch in (0, 5):
        trainer = make_trainer(current_epoch=epoch)
        for i in range(2):
            cb.on_validation_batch_end(trainer, model, None, make_batch(), i)
        with patch.object(model, "log_dict") as mock_log:
            cb.on_validation_epoch_end(trainer, model)
        log_call_count += mock_log.call_count

    assert log_call_count == 2
