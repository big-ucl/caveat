import numpy as np
import pandas as pd
import pytest
import torch

from caveat.encoding import sequence as seq


@pytest.mark.parametrize(
    "acts,durations,expected",
    [
        (
            [2, 3, 2],
            [0.3, 0.2, 0.5],
            np.array(
                [[0, 0.0], [2, 0.3], [3, 0.2], [2, 0.5], [1, 0.0], [1, 0.0]],
                dtype=np.float32,
            ),
        )
    ],
)
def test_encode_sequence(acts, durations, expected):
    encoded_sequence = seq.encode_sequence(
        acts, durations, max_length=6, encoding_width=2, sos=0, eos=1
    )
    np.testing.assert_array_equal(encoded_sequence, expected)


def test_encoder():
    schedules = pd.DataFrame(
        [
            [0, 0, 0, 4, 4],
            [0, 1, 4, 8, 4],
            [0, 0, 8, 10, 2],
            [1, 0, 0, 3, 3],
            [1, 1, 3, 7, 4],
            [1, 0, 7, 10, 3],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    length = 6
    duration = 10
    expected = torch.tensor(
        [[0, 0.0], [2, 0.4], [3, 0.4], [2, 0.2], [1, 0.0], [1, 0.0]]
    )
    expected_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    labels = torch.tensor([[0, 0], [1, 1]])
    label_weights = (torch.tensor([[1, 1], [1, 1]]), torch.tensor([[1], [1]]))
    encoder = seq.ContinuousEncoder(
        max_length=length,
        norm_duration=duration,
        weighting="unit",
        joint_weighting="unit",
    )
    encoded_data = encoder.encode(schedules, labels, label_weights)
    encoded_schedule = encoded_data.schedules
    masks = encoded_data.act_weights
    labels = encoded_data.labels
    labels_weights = encoded_data.label_weights

    assert torch.equal(encoded_schedule[0], expected)
    assert torch.equal(masks[0], expected_weights)
    assert torch.equal(labels[0], torch.tensor([0, 0]))
    assert torch.equal(labels_weights[0], torch.tensor([1, 1]))


def test_norm_durations():
    data = pd.DataFrame(
        [
            [0, 0, 0, 2, 2],
            [0, 1, 2, 4, 2],
            [0, 0, 4, 5, 1],
            [1, 0, 0, 6, 6],
            [1, 1, 6, 12, 6],
            [1, 0, 12, 15, 3],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    expected = pd.DataFrame(
        [
            [0, 0, 0, 4, 4],
            [0, 1, 4, 8, 4],
            [0, 0, 8, 10, 2],
            [1, 0, 0, 4, 4],
            [1, 1, 4, 8, 4],
            [1, 0, 8, 10, 2],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    result = seq.norm_durations(data, 10)
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_fix_end_durations():
    data = pd.DataFrame(
        [
            [0, 0, 0, 4],
            [0, 1, 4, 8],
            [0, 0, 8, 11],
            [1, 0, 0, 4],
            [1, 1, 4, 8],
            [1, 0, 8, 10],
        ],
        columns=["pid", "act", "start", "end"],
    )
    expected = pd.DataFrame(
        [
            [0, 0, 0, 4],
            [0, 1, 4, 8],
            [0, 0, 8, 10],
            [1, 0, 0, 4],
            [1, 1, 4, 8],
            [1, 0, 8, 10],
        ],
        columns=["pid", "act", "start", "end"],
    )
    result = seq.fix_end_durations(data, 10)
    pd.testing.assert_frame_equal(result, expected)


def test_decode():
    duration = 10
    encoded = torch.tensor(
        [
            [[0, 0.0], [2, 0.4], [3, 0.4], [2, 0.2], [1, 0.0], [1, 0.0]],
            [[0, 0.0], [2, 0.3], [3, 0.4], [2, 0.3], [1, 0.0], [1, 0.0]],
        ]
    )
    expected = pd.DataFrame(
        [
            [0, 0, 0, 4, 4],
            [0, 1, 4, 8, 4],
            [0, 0, 8, 10, 2],
            [1, 0, 0, 3, 3],
            [1, 1, 3, 7, 4],
            [1, 0, 7, 10, 3],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    encoder = seq.ContinuousEncoder(max_length=6, norm_duration=duration)
    encoder.setup_encoder(expected)
    decoded = encoder.decode(encoded, argmax=False)
    pd.testing.assert_frame_equal(decoded, expected)


def test_decode_argmax():
    duration = 10
    encoded = torch.tensor(
        [
            [
                [0.9, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.8, 0.2, 0.4],
                [0.0, 0.0, 0.0, 0.7, 0.4],
                [0.0, 0.0, 0.3, 0.0, 0.2],
                [0.1, 0.9, 0.0, 0.0, 0.0],
                [0.0, 0.9, 0.0, 0.0, 0.0],
            ],
            [
                [0.9, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.8, 0.2, 0.3],
                [0.0, 0.0, 0.0, 0.7, 0.4],
                [0.0, 0.0, 0.3, 0.0, 0.3],
                [0.1, 0.9, 0.0, 0.0, 0.0],
                [0.0, 0.9, 0.0, 0.0, 0.0],
            ],
        ]
    )
    expected = pd.DataFrame(
        [
            [0, 0, 0, 4, 4],
            [0, 1, 4, 8, 4],
            [0, 0, 8, 10, 2],
            [1, 0, 0, 3, 3],
            [1, 1, 3, 7, 4],
            [1, 0, 7, 10, 3],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    encoder = seq.ContinuousEncoder(max_length=6, norm_duration=duration)
    encoder.setup_encoder(expected)
    decoded = encoder.decode(encoded, argmax=True)
    pd.testing.assert_frame_equal(decoded, expected)


def test_decode_fix_durations():
    duration = 10
    encoded = torch.tensor(
        [
            [[0, 0.0], [2, 0.2], [3, 0.2], [2, 0.1], [1, 0.0], [1, 0.0]],
            [[0, 0.0], [2, 0.6], [3, 0.8], [2, 0.6], [1, 0.0], [1, 0.0]],
        ]
    )
    expected = pd.DataFrame(
        [
            [0, 0, 0, 4, 4],
            [0, 1, 4, 8, 4],
            [0, 0, 8, 10, 2],
            [1, 0, 0, 3, 3],
            [1, 1, 3, 7, 4],
            [1, 0, 7, 10, 3],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    encoder = seq.ContinuousEncoder(
        max_length=6, norm_duration=duration, fix_durations=True
    )
    encoder.setup_encoder(expected)
    decoded = encoder.decode(encoded, argmax=False)
    pd.testing.assert_frame_equal(decoded, expected)
