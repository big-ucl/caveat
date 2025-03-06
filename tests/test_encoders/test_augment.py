from torch import allclose, tensor

from caveat.data.augment import DiscreteJitter, SequenceJitter


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
