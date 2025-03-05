import numpy as np
import pandas as pd
import pytest
import torch
from torch import tensor

from caveat.encoding import discrete


@pytest.mark.parametrize(
    "acts,starts,ends,length,expected",
    [
        ([0, 1, 0], [0, 2, 5], [2, 5, 6], 6, np.array([0, 0, 1, 1, 1, 0])),
        ([0, 1, 0], [0, 2, 5], [2, 5, 6], 5, np.array([0, 0, 1, 1, 1])),
        ([1, 0], [2, 5], [5, 6], 6, np.array([0, 0, 1, 1, 1, 0])),
    ],
)
def test_discretise_trace(acts, starts, ends, length, expected):
    result = discrete.discretise_trace(acts, starts, ends, length)
    np.testing.assert_array_equal(result, expected)


def test_discretise_population():
    traces = pd.DataFrame(
        [
            [0, 0, 0, 2],
            [0, 1, 2, 5],
            [0, 0, 5, 6],
            [1, 0, 0, 3],
            [1, 1, 3, 5],
            [1, 0, 5, 6],
        ],
        columns=["pid", "act", "start", "end"],
    )
    length = 6
    step = 2
    expected = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int8)
    result = discrete.discretise_population(traces, length, step)
    expected = torch.from_numpy(expected)
    assert torch.equal(result, expected)


def test_large_downsample_discretise_population():
    traces = pd.DataFrame(
        [[0, 0, 0, 360], [0, 1, 360, 840], [0, 0, 840, 1440]],
        columns=["pid", "act", "start", "end"],
    )
    length = 1440
    step = 180
    expected = np.array([[0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.int8)
    result = discrete.discretise_population(traces, length, step)
    expected = torch.from_numpy(expected)
    assert torch.equal(result, expected)


def test_encoded_weights():
    traces = pd.DataFrame(
        [
            [0, 0, 0, 2, 2],
            [0, 1, 2, 5, 3],
            [0, 0, 5, 6, 1],
            [1, 0, 0, 3, 3],
            [1, 1, 3, 5, 2],
            [1, 0, 5, 6, 1],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    length = 6
    step = 2
    expected_weights = torch.tensor([7.0, 5.0])
    expected_weights = 1 / torch.log(expected_weights)
    expected_weights = expected_weights / expected_weights.mean()
    steps = length / step
    expected_weights = expected_weights / steps
    expected_mask = torch.tensor([1.0, 1.0, 1.0])
    encoder = discrete.DiscreteEncoder(length, step)
    encoded = encoder.encode(traces, None, (None, None))
    assert torch.equal(encoded.encoding_weights, expected_weights)
    assert torch.equal(encoded[0][0][1][0], expected_mask)


def test_encoded_with_jitter():
    traces = pd.DataFrame(
        [
            [0, 0, 0, 2, 2],
            [0, 1, 2, 5, 3],
            [0, 0, 5, 6, 1],
            [1, 0, 0, 3, 3],
            [1, 1, 3, 5, 2],
            [1, 0, 5, 6, 1],
        ],
        columns=["pid", "act", "start", "end", "duration"],
    )
    length = 6
    step = 2
    encoder = discrete.DiscreteEncoder(length, step, jitter=0.2)
    encoded = encoder.encode(traces, None, (None, None))
    for _ in range(10):
        for i in range(len(encoded)):
            (left, _), (right, _), _ = encoded[i]
            assert left.shape == (3,)
            assert torch.equal(left, right)


def test_padded_encoder():
    traces = pd.DataFrame(
        [
            [0, 0, 0, 2],
            [0, 1, 2, 5],
            [0, 0, 5, 6],
            [1, 0, 0, 3],
            [1, 1, 3, 5],
            [1, 0, 5, 6],
        ],
        columns=["pid", "act", "start", "end"],
    )
    traces["duration"] = traces.end - traces.start
    length = 6
    step = 2
    encode = torch.tensor([[1, 2, 2], [1, 1, 2]])
    pad_left = torch.tensor([[0, 1, 2, 2], [0, 1, 1, 2]])
    pad_right = torch.tensor([[1, 2, 2, 0], [1, 1, 2, 0]])
    masks = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0])
    weights = torch.tensor([1 / 120, 1 / 7, 1 / 5])
    encoder = discrete.DiscreteEncoderPadded(
        duration=length, step_size=step, jitter=0
    )
    encoded = encoder.encode(traces, None)
    assert torch.equal(encoded.schedules, encode)
    assert torch.equal(encoded.act_weights, masks)
    assert torch.equal(encoded.encoding_weights, weights)
    (left, (mask_left, _)), (right, (mask_right, _)), _ = encoded[0]
    assert torch.equal(left, pad_left[0])
    assert torch.equal(mask_left, mask)
    assert torch.equal(right, pad_right[0])
    assert torch.equal(mask_right, mask)


def test_padded_encoder_with_jitter():
    traces = pd.DataFrame(
        [
            [0, 0, 0, 2],
            [0, 1, 2, 5],
            [0, 0, 5, 6],
            [1, 0, 0, 3],
            [1, 1, 3, 5],
            [1, 0, 5, 6],
        ],
        columns=["pid", "act", "start", "end"],
    )
    traces["duration"] = traces.end - traces.start
    length = 6
    step = 2
    encoder = discrete.DiscreteEncoderPadded(
        duration=length, step_size=step, jitter=0.2
    )
    encoded = encoder.encode(traces, None)
    for _ in range(10):
        for i in range(len(encoded)):
            (left, _), (right, _), _ = encoded[i]
            assert left.shape == (4,)
            assert torch.equal(left[1:], right[:-1])


@pytest.mark.parametrize(
    "encoded,length,step_size,expected",
    [
        (
            tensor(
                [
                    [[0.8, 0.1, 0.1], [0.7, 0.0, 0.0], [0.1, 0.5, 0.1]],
                    [[1, 0, 0], [0, 0, 1], [1, 0, 0]],
                ]
            ),
            144,
            48,
            pd.DataFrame(
                [
                    [0, "a", 0, 96],
                    [0, "b", 96, 144],
                    [1, "a", 0, 48],
                    [1, "c", 48, 96],
                    [1, "a", 96, 144],
                ],
                columns=["pid", "act", "start", "end"],
            ),
        ),
        (
            tensor([[[1, 0, 0], [1, 0, 0], [0, 1, 0]]]),
            3,
            1,
            pd.DataFrame(
                [[0, "a", 0, 2], [0, "b", 2, 3]],
                columns=["pid", "act", "start", "end"],
            ),
        ),
    ],
)
def test_decode_discretised(encoded, length, step_size, expected):
    encoder = discrete.DiscreteEncoder(duration=length, step_size=step_size)
    encoder.index_to_acts = {0: "a", 1: "b", 2: "c"}
    result = encoder.decode(encoded)
    pd.testing.assert_frame_equal(expected, result)


@pytest.mark.parametrize(
    "encoded,length,step_size,expected",
    [
        (
            tensor(
                [
                    [
                        [0.1, 0.7, 0.1, 0.1],
                        [0.1, 0.7, 0.1, 0.1],
                        [0.3, 0.0, 0.7, 0.0],
                        [0.8, 0.1, 0.1, 0.1],
                    ],
                    [[0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
                ]
            ),
            144,
            48,
            pd.DataFrame(
                [
                    [0, "a", 0, 96],
                    [0, "b", 96, 144],
                    [1, "a", 0, 48],
                    [1, "c", 48, 96],
                    [1, "a", 96, 144],
                ],
                columns=["pid", "act", "start", "end"],
            ),
        ),
        (
            tensor([[[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]]),
            3,
            1,
            pd.DataFrame(
                [[0, "a", 0, 2], [0, "b", 2, 3]],
                columns=["pid", "act", "start", "end"],
            ),
        ),
    ],
)
def test_decode_padded(encoded, length, step_size, expected):
    encoder = discrete.DiscreteEncoderPadded(
        duration=length, step_size=step_size
    )
    encoder.index_to_acts = {0: "<PAD>", 1: "a", 2: "b", 3: "c"}
    result = encoder.decode(encoded)
    pd.testing.assert_frame_equal(expected, result)
