import pytest
from torch import allclose, tensor

from caveat.data.augment import (
    DiscreteJitter,
    DiscreteSingleJitter,
    ScheduleAugment,
    SequenceJitter,
    SmallestJitter,
)


def test_sequence_jitter_zero():
    jitterer = SequenceJitter(jitter=0)
    seq = tensor([[0, 0], [2, 0.3], [3, 0.5], [2, 0.2], [1, 0], [1, 0]])
    out = jitterer(seq)
    assert (seq == out).all()


def test_sequence_jitter():
    for j in [0.01, 0.1, 0.5]:
        jitterer = SequenceJitter(jitter=j)
        seq = tensor([[0, 0], [2, 0.3], [3, 0.5], [2, 0.2], [1, 0], [1, 0]])
        target_durations = tensor([0.3, 0.5, 0.2])
        zero = tensor(0.0)
        for _ in range(100):
            out = jitterer(seq)
            diff = seq[:, 1] - out[:, 1]
            assert allclose(diff.sum(), zero, atol=1e-6)
            assert diff[0] == 0
            assert diff[-1] == 0
            assert diff[-2] == 0
            abs_diff = diff.abs()[1:-2]
            rel_diff = abs_diff / target_durations
            assert rel_diff.sum() > 0
            assert rel_diff.max() <= j


def test_smallest_jitter():
    for j in [0.01, 0.1, 0.5]:
        jitterer = SequenceJitter(jitter=j)
        seq = tensor([[0, 0], [2, 0.3], [3, 0.5], [2, 0.2], [1, 0], [1, 0]])
        target_durations = tensor([0.3, 0.5, 0.2])
        zero = tensor(0.0)
        for _ in range(100):
            out = jitterer(seq)
            diff = seq[:, 1] - out[:, 1]
            assert allclose(diff.sum(), zero, atol=1e-6)
            assert diff[0] == 0
            assert diff[-1] == 0
            assert diff[-2] == 0
            abs_diff = diff.abs()[1:-2]
            rel_diff = abs_diff / target_durations
            assert rel_diff.sum() > 0
            assert rel_diff.max() <= j


def test_discrete_jitter_zero():
    jitterer = DiscreteJitter(step_size=144, jitter=0)
    seq = tensor([0, 0, 1, 1, 2, 2, 1, 0, 0, 0])
    out = jitterer(seq)
    assert (seq == out).all()


def test_discrete_jitter():
    meta_diffs = []
    for j in [0.1, 0.5]:
        jitterer = DiscreteJitter(step_size=30, jitter=j)
        seq = tensor(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        diffs = []
        for _ in range(100):
            out = jitterer(seq)
            diff = abs(seq - out)
            changes = diff > 0
            diffs.append(changes.sum())
        assert sum(diffs) > 0
        assert max(diffs) <= 28 * j
        meta_diffs.append(sum(diffs))
    assert meta_diffs[0] < meta_diffs[1]


# --- ScheduleAugment (abstract base) ---


def test_schedule_augment_init_raises():
    with pytest.raises(NotImplementedError):
        ScheduleAugment()


def test_schedule_augment_call_raises():
    class _Impl(ScheduleAugment):
        def __init__(self):
            pass

    with pytest.raises(NotImplementedError):
        _Impl()(tensor([1, 2, 3]))


# --- SequenceJitter: single-activity edge case ---


def test_sequence_jitter_single_activity():
    jitterer = SequenceJitter(jitter=0.5)
    # only one real activity (mask = seq[:, 0] > 1 has sum == 1)
    seq = tensor([[0.0, 0.0], [2.0, 0.5], [1.0, 0.0]])
    out = jitterer(seq)
    assert (seq == out).all()


# --- SmallestJitter ---


def test_smallest_jitter_zero():
    jitterer = SmallestJitter(jitter=0)
    seq = tensor([[0.0, 0.0], [2.0, 0.3], [3.0, 0.5], [1.0, 0.0]])
    out = jitterer(seq)
    assert (seq == out).all()


def test_smallest_jitter_single_activity():
    jitterer = SmallestJitter(jitter=0.5)
    # mask.sum() == 1 → early return
    seq = tensor([[0.0, 0.0], [2.0, 0.5], [1.0, 0.0]])
    out = jitterer(seq)
    assert (seq == out).all()


def test_smallest_jitter_multi_activity():
    zero = tensor(0.0)
    for j in [0.01, 0.1, 0.5]:
        jitterer = SmallestJitter(jitter=j)
        seq = tensor(
            [
                [0.0, 0.0],
                [2.0, 0.3],
                [3.0, 0.5],
                [2.0, 0.2],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        for _ in range(100):
            out = jitterer(seq)
            diff = seq[:, 1] - out[:, 1]
            assert allclose(diff.sum(), zero, atol=1e-6)
            assert diff[0] == 0
            assert diff[-1] == 0
            assert diff[-2] == 0


# --- DiscreteSingleJitter ---


def test_discrete_single_jitter_init():
    jitterer = DiscreteSingleJitter(step_size=30, jitter=2)
    assert jitterer.step_size == 30
    assert jitterer.jitter == 2


def test_discrete_single_jitter_no_transitions():
    jitterer = DiscreteSingleJitter(step_size=30, jitter=1)
    seq = tensor([1, 1, 1, 1, 1])
    out = jitterer(seq)
    assert (seq == out).all()


def test_discrete_single_jitter_zero():
    jitterer = DiscreteSingleJitter(step_size=30, jitter=0)
    seq = tensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
    out = jitterer(seq)
    assert (seq == out).all()


def test_discrete_single_jitter_changes():
    jitterer = DiscreteSingleJitter(step_size=30, jitter=1)
    seq = tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    changed = any(not (seq == jitterer(seq)).all() for _ in range(100))
    assert changed


def test_discrete_single_jitter_exhaustive():
    # run many times to exercise both direction branches deterministically
    import numpy as np

    np.random.seed(0)
    jitterer = DiscreteSingleJitter(step_size=30, jitter=1)
    seq = tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    for _ in range(500):
        jitterer(seq)


# --- DiscreteJitter: no-transitions edge case (line 125) ---


def test_discrete_jitter_no_transitions():
    jitterer = DiscreteJitter(step_size=30, jitter=1)
    seq = tensor([2, 2, 2, 2, 2, 2])
    out = jitterer(seq)
    assert (seq == out).all()
