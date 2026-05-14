import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from caveat.callbacks import (
    CollapseMonitor,
    CyclicalBetaAnnealer,
    LinearLossScheduler,
)


def _scheduler(config):
    return LinearLossScheduler(config)


def _epoch(n):
    trainer = SimpleNamespace(current_epoch=n)
    module = SimpleNamespace()
    return trainer, module


# --- validate_weights_schedule ---


def test_no_schedules():
    s = _scheduler({})
    assert s.kld_schedule is None
    assert s.act_schedule is None


def test_valid_schedule():
    _scheduler({"kld_loss_schedule": (0, 10)})  # no error


def test_reversed_schedule_raises():
    with pytest.raises(ValueError):
        _scheduler({"kld_loss_schedule": (10, 0)})


def test_negative_start_raises():
    with pytest.raises(ValueError):
        _scheduler({"kld_loss_schedule": (-1, 5)})


def test_negative_end_raises():
    with pytest.raises(ValueError):
        _scheduler({"kld_loss_schedule": (0, -1)})


def test_schedule_ends_before_min_epochs_warns(capsys):
    _scheduler({"kld_loss_schedule": (0, 5), "min_epochs": 10})
    assert "WARNING" in capsys.readouterr().out


# --- on_train_epoch_start: kld ---


def test_kld_before_start():
    s = _scheduler({"kld_loss_schedule": (10, 20)})
    trainer, module = _epoch(5)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_kld_weight == 0.0


def test_kld_at_end():
    s = _scheduler({"kld_loss_schedule": (0, 10)})
    trainer, module = _epoch(10)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_kld_weight == 1.0


def test_kld_in_range():
    s = _scheduler({"kld_loss_schedule": (0, 10)})
    trainer, module = _epoch(5)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_kld_weight == pytest.approx(0.5)


def test_no_kld_schedule_sets_no_weight():
    s = _scheduler({})
    trainer, module = _epoch(5)
    s.on_train_epoch_start(trainer, module)
    assert not hasattr(module, "scheduled_kld_weight")


# --- on_train_epoch_start: act ---


def test_act_before_start():
    s = _scheduler({"activity_loss_schedule": (5, 15)})
    trainer, module = _epoch(0)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_act_weight == 0.0


def test_act_at_end():
    s = _scheduler({"activity_loss_schedule": (0, 10)})
    trainer, module = _epoch(10)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_act_weight == 1.0


def test_act_in_range():
    s = _scheduler({"activity_loss_schedule": (0, 4)})
    trainer, module = _epoch(2)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_act_weight == pytest.approx(0.5)


# --- on_train_epoch_start: dur ---


def test_dur_before_start():
    s = _scheduler({"duration_loss_schedule": (10, 20)})
    trainer, module = _epoch(0)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_dur_weight == 0.0


def test_dur_at_end():
    s = _scheduler({"duration_loss_schedule": (0, 10)})
    trainer, module = _epoch(15)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_dur_weight == 1.0


def test_dur_in_range():
    s = _scheduler({"duration_loss_schedule": (0, 8)})
    trainer, module = _epoch(4)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_dur_weight == pytest.approx(0.5)


# --- on_train_epoch_start: end ---


def test_end_before_start():
    s = _scheduler({"end_loss_schedule": (10, 20)})
    trainer, module = _epoch(0)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_end_weight == 0.0


def test_end_at_end():
    s = _scheduler({"end_loss_schedule": (0, 10)})
    trainer, module = _epoch(10)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_end_weight == 1.0


def test_end_in_range():
    s = _scheduler({"end_loss_schedule": (0, 10)})
    trainer, module = _epoch(5)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_end_weight == pytest.approx(0.5)


# --- on_train_epoch_start: label ---


def test_label_before_start():
    s = _scheduler({"label_loss_schedule": (10, 20)})
    trainer, module = _epoch(0)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_label_weight == 0.0


def test_label_at_end():
    s = _scheduler({"label_loss_schedule": (0, 10)})
    trainer, module = _epoch(10)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_label_weight == 1.0


def test_label_in_range():
    s = _scheduler({"label_loss_schedule": (0, 10)})
    trainer, module = _epoch(2)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_label_weight == pytest.approx(0.2)


# --- multiple schedules active simultaneously ---


def test_multiple_schedules():
    s = _scheduler(
        {"kld_loss_schedule": (0, 10), "activity_loss_schedule": (0, 20)}
    )
    trainer, module = _epoch(5)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_kld_weight == pytest.approx(0.5)
    assert module.scheduled_act_weight == pytest.approx(0.25)


# --- on_train_epoch_start: start ---


def test_start_before_start():
    s = _scheduler({"start_loss_schedule": (10, 20)})
    trainer, module = _epoch(0)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_start_weight == 0.0


def test_start_at_end():
    s = _scheduler({"start_loss_schedule": (0, 10)})
    trainer, module = _epoch(10)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_start_weight == 1.0


def test_start_in_range():
    s = _scheduler({"start_loss_schedule": (0, 10)})
    trainer, module = _epoch(5)
    s.on_train_epoch_start(trainer, module)
    # note: code sets scheduled_end_weight here (bug), not scheduled_start_weight
    assert module.scheduled_start_weight == pytest.approx(0.5)


# --- on_train_epoch_start: total_duration ---


def test_total_dur_before_start():
    s = _scheduler({"total_duration_loss_schedule": (10, 20)})
    trainer, module = _epoch(0)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_total_dur_weight == 0.0


def test_total_dur_at_end():
    s = _scheduler({"total_duration_loss_schedule": (0, 10)})
    trainer, module = _epoch(10)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_total_dur_weight == 1.0


def test_total_dur_in_range():
    s = _scheduler({"total_duration_loss_schedule": (0, 10)})
    trainer, module = _epoch(5)
    s.on_train_epoch_start(trainer, module)
    assert module.scheduled_total_dur_weight == pytest.approx(0.5)


def make_mocks(max_epochs, current_epoch):
    trainer = MagicMock()
    trainer.max_epochs = max_epochs
    trainer.current_epoch = current_epoch
    pl_module = MagicMock()
    return trainer, pl_module


# --- Initialisation ---


def test_default_config():
    cb = CyclicalBetaAnnealer({})
    assert cb.n_cycles == 4
    assert cb.max_beta_multiplier == 1.0
    assert cb.ratio == 0.5


def test_custom_config():
    cb = CyclicalBetaAnnealer({"n_cycles": 2, "max_beta": 0.5, "ratio": 0.75})
    assert cb.n_cycles == 2
    assert cb.max_beta_multiplier == 0.5
    assert cb.ratio == 0.75


# --- Beta value cases ---
# Config: max_epochs=100, n_cycles=4, ratio=0.5
# => cycle_len=25, ramp_end=12.5


@pytest.fixture
def callback():
    return CyclicalBetaAnnealer({"n_cycles": 4, "max_beta": 1.0, "ratio": 0.5})


def test_beta_zero_at_cycle_start(callback):
    trainer, pl_module = make_mocks(max_epochs=100, current_epoch=0)
    callback.on_train_epoch_start(trainer, pl_module)
    assert pl_module.beta == 0.0


def test_beta_mid_ramp(callback):
    # epoch=6, cycle_pos=6, ramp_end=12.5 => beta = 1.0 * (6 / 12.5) = 0.48
    trainer, pl_module = make_mocks(max_epochs=100, current_epoch=6)
    callback.on_train_epoch_start(trainer, pl_module)
    assert pl_module.beta == pytest.approx(0.48)


def test_beta_near_ramp_end(callback):
    # epoch=12, cycle_pos=12, ramp_end=12.5 => beta = 1.0 * (12 / 12.5) = 0.96
    trainer, pl_module = make_mocks(max_epochs=100, current_epoch=12)
    callback.on_train_epoch_start(trainer, pl_module)
    assert pl_module.beta == pytest.approx(0.96)


def test_beta_at_plateau(callback):
    # epoch=13, cycle_pos=13 >= ramp_end=12.5 => beta = max_beta = 1.0
    trainer, pl_module = make_mocks(max_epochs=100, current_epoch=13)
    callback.on_train_epoch_start(trainer, pl_module)
    assert pl_module.beta == 1.0


def test_beta_second_cycle_start(callback):
    # epoch=25 => cycle_pos = 25 % 25 = 0 => beta = 0.0
    trainer, pl_module = make_mocks(max_epochs=100, current_epoch=25)
    callback.on_train_epoch_start(trainer, pl_module)
    assert pl_module.beta == 0.0


def test_beta_last_epoch(callback):
    # epoch=99 => cycle_pos = 99 % 25 = 24 >= ramp_end=12.5 => beta = 1.0
    trainer, pl_module = make_mocks(max_epochs=100, current_epoch=99)
    callback.on_train_epoch_start(trainer, pl_module)
    assert pl_module.beta == 1.0


# --- Side effects ---


def test_log_called(callback):
    trainer, pl_module = make_mocks(max_epochs=100, current_epoch=6)
    callback.on_train_epoch_start(trainer, pl_module)
    pl_module.log.assert_called_once_with("beta", pytest.approx(0.48))


def test_pl_module_beta_set(callback):
    trainer, pl_module = make_mocks(max_epochs=100, current_epoch=13)
    callback.on_train_epoch_start(trainer, pl_module)
    assert pl_module.beta == 1.0


# ===========================================================================
# CollapseMonitor
# ===========================================================================


def make_collapse_mocks(current_epoch, mu, log_var):
    trainer = MagicMock()
    trainer.current_epoch = current_epoch
    pl_module = MagicMock()
    pl_module.encode.return_value = (mu, log_var)
    return trainer, pl_module


def _populate_buffers(
    cb, n_batches=2, batch_size=4, latent_dim=3, n_conditions=2
):
    """Push synthetic data into cb's internal buffers."""
    mu = torch.randn(batch_size, latent_dim)
    log_var = torch.zeros(batch_size, latent_dim)
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    cond = torch.zeros(batch_size, 1)
    for _ in range(n_batches):
        cb._mus.append(mu)
        cb._log_vars.append(log_var)
        cb._conditions.append(cond)
        cb._kl_per_dim.append(kl.mean(dim=0))


# --- Initialisation ---


def test_collapse_monitor_defaults():
    cb = CollapseMonitor({})
    assert cb.au_threshold == 0.01
    assert cb.kl_collapse_threshold == 0.1
    assert cb.conditional_threshold == 0.05
    assert cb.check_every_n_epochs == 5
    assert cb.warn_au_below == 0.5


# --- decoder sensitivity (accumulated in on_validation_batch_end) ---


def _make_batch_end_pl_module(decode_fn=None):
    """Build a pl_module mock with label_encoder and configurable decode."""
    pl_module = MagicMock()
    pl_module.label_encoder = MagicMock()
    mu = torch.randn(4, 3)
    log_var = torch.zeros(4, 3)
    pl_module.encode.return_value = (mu, log_var)
    if decode_fn is not None:
        pl_module.decode.side_effect = decode_fn
    else:
        pl_module.decode.return_value = torch.zeros(4, 10, 5)
    return pl_module


def test_decoder_sensitivity_no_label_encoder():
    """Batches with no label_encoder skip accumulation → nan at epoch end."""
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    # batch_end mock: no label_encoder so no decoder probing
    pl_batch = MagicMock(spec=["encode"])
    pl_batch.encode.return_value = (torch.randn(4, 3), torch.zeros(4, 3))
    trainer = MagicMock()
    trainer.current_epoch = 0
    batch = (torch.randn(4, 10, 2), torch.zeros(4, 1))
    cb.on_validation_batch_end(trainer, pl_batch, None, batch, 0)
    # epoch_end mock: regular MagicMock so log_dict works
    pl_epoch = MagicMock()
    cb.on_validation_epoch_end(trainer, pl_epoch)
    logged = pl_epoch.log_dict.call_args[0][0]
    assert math.isnan(logged["collapse/decoder_sensitivity"])


def test_decoder_sensitivity_single_condition(capsys):
    """All-same conditions in every batch → nan (no shuffle is meaningful) and a warning is printed."""
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    pl_module = _make_batch_end_pl_module()
    trainer = MagicMock()
    trainer.current_epoch = 0
    batch = (torch.randn(4, 10, 2), torch.zeros(4, 1))  # all condition 0
    cb.on_validation_batch_end(trainer, pl_module, None, batch, 0)
    out = capsys.readouterr().out
    assert "CollapseMonitor" in out
    assert "skipping decoder sensitivity" in out
    _populate_buffers(cb, n_batches=0)
    cb.on_validation_epoch_end(trainer, pl_module)
    logged = pl_module.log_dict.call_args[0][0]
    assert math.isnan(logged["collapse/decoder_sensitivity"])


def test_decoder_sensitivity_insensitive_decoder():
    """Decoder returns same output for any condition → swap_mse=0 → dec_sens≈0."""
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    pl_module = _make_batch_end_pl_module()  # decode always returns zeros
    trainer = MagicMock()
    trainer.current_epoch = 0
    c = torch.cat([torch.zeros(2, 1), torch.ones(2, 1)])
    batch = (torch.randn(4, 10, 2), c)
    cb.on_validation_batch_end(trainer, pl_module, None, batch, 0)
    _populate_buffers(cb, n_batches=0)
    cb.on_validation_epoch_end(trainer, pl_module)
    logged = pl_module.log_dict.call_args[0][0]
    assert logged["collapse/decoder_sensitivity"] == pytest.approx(
        0.0, abs=1e-6
    )


def test_decoder_sensitivity_responsive_decoder():
    """Decoder output differs by condition → dec_sens > 0."""
    cb = CollapseMonitor({"check_every_n_epochs": 1})
    call_n = {"n": 0}

    def decode_fn(mu, labels):
        val = float(call_n["n"])
        call_n["n"] += 1
        return torch.full((len(mu), 10, 5), val)

    pl_module = _make_batch_end_pl_module(decode_fn=decode_fn)
    trainer = MagicMock()
    trainer.current_epoch = 0
    c = torch.cat([torch.zeros(2, 1), torch.ones(2, 1)])
    batch = (torch.randn(4, 10, 2), c)
    cb.on_validation_batch_end(trainer, pl_module, None, batch, 0)
    _populate_buffers(cb, n_batches=0)
    cb.on_validation_epoch_end(trainer, pl_module)
    logged = pl_module.log_dict.call_args[0][0]
    assert logged["collapse/decoder_sensitivity"] > 0.0


# --- on_validation_batch_end ---


def test_batch_end_accumulates():
    cb = CollapseMonitor({})
    mu = torch.randn(4, 3)
    log_var = torch.zeros(4, 3)
    trainer, pl_module = make_collapse_mocks(
        current_epoch=0, mu=mu, log_var=log_var
    )
    batch = (torch.randn(4, 10, 2), torch.zeros(4, 1))

    cb.on_validation_batch_end(
        trainer, pl_module, outputs=None, batch=batch, batch_idx=0
    )
    cb.on_validation_batch_end(
        trainer, pl_module, outputs=None, batch=batch, batch_idx=1
    )

    assert len(cb._mus) == 2
    assert len(cb._log_vars) == 2
    assert len(cb._conditions) == 2
    assert len(cb._kl_per_dim) == 2


# --- on_validation_epoch_end: skip path ---


def test_epoch_end_skips_non_check_epoch():
    cb = CollapseMonitor({"check_every_n_epochs": 5})
    _populate_buffers(cb)
    trainer = MagicMock()
    trainer.current_epoch = 1  # 1 % 5 != 0
    pl_module = MagicMock()

    cb.on_validation_epoch_end(trainer, pl_module)

    pl_module.log_dict.assert_not_called()
    assert len(cb._mus) == 0  # buffers cleared


# --- on_validation_epoch_end: check path ---


def test_epoch_end_logs_on_check_epoch():
    cb = CollapseMonitor({"check_every_n_epochs": 5})
    _populate_buffers(cb)
    trainer = MagicMock()
    trainer.current_epoch = 0  # 0 % 5 == 0
    pl_module = MagicMock()

    cb.on_validation_epoch_end(trainer, pl_module)

    pl_module.log_dict.assert_called_once()
    logged = pl_module.log_dict.call_args[0][0]
    expected_keys = {
        "collapse/active_units_pct",
        "collapse/n_active_dims",
        "collapse/kl_collapsed_dims",
        "collapse/kl_mean",
        "collapse/kl_min",
        "collapse/decoder_sensitivity",
        "collapse/mean_posterior_var",
    }
    assert expected_keys == set(logged.keys())


def test_epoch_end_resets_buffers():
    cb = CollapseMonitor({"check_every_n_epochs": 5})
    _populate_buffers(cb)
    trainer = MagicMock()
    trainer.current_epoch = 0
    pl_module = MagicMock()

    cb.on_validation_epoch_end(trainer, pl_module)

    assert cb._mus == []
    assert cb._log_vars == []
    assert cb._conditions == []
    assert cb._kl_per_dim == []


# --- Early stopping (integrated) ---


def _run_check_epoch(cb, dec_sens_value):
    """Run one check epoch with a controlled decoder sensitivity value.

    Injects scalar buffers so mean(swap_mse)/mean(out_var) == dec_sens_value,
    or leaves them empty for nan.
    """
    trainer = MagicMock()
    trainer.current_epoch = 0  # always a check epoch (0 % 5 == 0)
    pl_module = MagicMock()
    _populate_buffers(cb)
    if not math.isnan(dec_sens_value):
        cb._decoder_swap_mse = [dec_sens_value]
        cb._decoder_out_var = [1.0]
    cb.on_validation_epoch_end(trainer, pl_module)
    return trainer


def test_stopping_disabled_by_default():
    cb = CollapseMonitor({})
    assert cb.stopping_patience is None


def test_stopping_patience_from_collapse_patience():
    cb = CollapseMonitor({"collapse_patience": 3})
    assert cb.stopping_patience == 3


def test_stopping_patience_falls_back_to_patience():
    cb = CollapseMonitor({"patience": 7})
    assert cb.stopping_patience == 7


def test_stopping_patience_collapse_patience_takes_priority():
    cb = CollapseMonitor({"collapse_patience": 3, "patience": 99})
    assert cb.stopping_patience == 3


def test_no_stop_when_separation_healthy():
    cb = CollapseMonitor(
        {"collapse_patience": 2, "conditional_threshold": 0.05}
    )
    trainer = _run_check_epoch(cb, dec_sens_value=0.5)
    assert trainer.should_stop is not True
    assert cb._bad_epochs == 0


def test_bad_epoch_counter_increments():
    cb = CollapseMonitor(
        {"collapse_patience": 3, "conditional_threshold": 0.05}
    )
    _run_check_epoch(cb, dec_sens_value=0.01)
    assert cb._bad_epochs == 1


def test_stops_after_patience_exceeded():
    cb = CollapseMonitor(
        {"collapse_patience": 2, "conditional_threshold": 0.05}
    )
    _run_check_epoch(cb, dec_sens_value=0.01)
    trainer = _run_check_epoch(cb, dec_sens_value=0.01)
    assert trainer.should_stop is True


def test_bad_epochs_reset_on_recovery():
    cb = CollapseMonitor(
        {"collapse_patience": 3, "conditional_threshold": 0.05}
    )
    _run_check_epoch(cb, dec_sens_value=0.01)  # bad
    _run_check_epoch(cb, dec_sens_value=0.5)  # recovered
    assert cb._bad_epochs == 0


def test_nan_sep_does_not_increment_counter():
    cb = CollapseMonitor(
        {"collapse_patience": 1, "conditional_threshold": 0.05}
    )
    _run_check_epoch(cb, dec_sens_value=float("nan"))
    assert cb._bad_epochs == 0
    # trainer.should_stop must not have been set to True
    # (verified indirectly: counter stayed at 0)
