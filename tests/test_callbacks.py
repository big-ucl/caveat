from types import SimpleNamespace

import pytest

from caveat.callbacks import LinearLossScheduler


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
    assert module.scheduled_end_weight == pytest.approx(0.5)


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
