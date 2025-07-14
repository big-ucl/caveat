import numpy as np
import pytest
from torch import equal, nn, randn, tensor

from caveat.models import utils


@pytest.mark.parametrize(
    "target,axis,expected",
    [
        (tensor([0.1, 0.2]), 0, tensor([0, 1])),
        (tensor([[0.1, 0.2]]), 1, tensor([[0, 1]])),
        (tensor([[0.1, 0.2]]), 0, tensor([[0, 1]])),
        (tensor([[0.1, 0.1]]), 1, tensor([[1, 0]])),
        (tensor([[0.1, 0.1]]), 0, tensor([[1, 0]])),
        (tensor([[0.1, 0.2], [0.2, 0.1]]), 1, tensor([[0, 1], [1, 0]])),
        (tensor([[0.1, 0.2], [0.2, 0.1]]), 0, tensor([[1, 0], [0, 1]])),
        (tensor([[[0.1, 0.2]], [[0.2, 0.1]]]), 2, tensor([[[0, 1]], [[1, 0]]])),
        (tensor([[[0.1, 0.2]], [[0.2, 0.1]]]), 1, tensor([[[1, 0]], [[0, 1]]])),
        (tensor([[[0.1, 0.2]], [[0.2, 0.1]]]), 0, tensor([[[1, 0]], [[0, 1]]])),
        (
            tensor([[[0.1, 0.2], [0.2, 0.1]], [[0.2, 0.1], [0.2, 0.1]]]),
            2,
            tensor([[[0, 1], [1, 0]], [[1, 0], [1, 0]]]),
        ),
        (
            tensor([[[0.1, 0.2], [0.2, 0.1]], [[0.2, 0.1], [0.2, 0.1]]]),
            1,
            tensor([[[0, 1], [1, 0]], [[1, 1], [0, 0]]]),
        ),
        (
            tensor([[[0.1, 0.2], [0.3, 0.1]], [[0.2, 0.1], [0.2, 0.15]]]),
            0,
            tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
        ),
    ],
)
def test_argmax_on_axis(target, axis, expected):
    result = utils.hot_argmax(target, axis)
    result = result.to(device="cpu")
    equal(result, expected)


@pytest.mark.parametrize(
    "size,kernel,stride,padding,dilation,expected",
    [
        (5, 3, 2, 1, 1, np.array([3, 3])),
        (5, 3, 1, 1, 1, np.array([5, 5])),
        (6, 3, 2, 1, 1, np.array([3, 3])),
        (144, 3, 2, 1, 1, np.array([72, 72])),
        (144, 3, 1, 1, 1, np.array([144, 144])),
        (np.array([5, 5]), 3, np.array([2, 1]), 1, 1, np.array([3, 5])),
    ],
)
def test_conv_size(size, kernel, stride, padding, dilation, expected):
    result = utils.conv2d_size(size, kernel, stride, padding, dilation)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "size,kernel,stride,padding,dilation,output_padding,expected",
    [
        (3, 3, 2, 1, 1, 0, np.array([5, 5])),
        (3, 3, 2, 1, 1, 1, np.array([6, 6])),
        (4, 3, 2, 1, 1, 0, np.array([7, 7])),
        (72, 3, 2, 1, 1, 0, np.array([143, 143])),
        (72, 3, 2, 1, 1, 1, np.array([144, 144])),
        (36, 3, 2, 1, 1, 0, np.array([71, 71])),
        (36, 3, 2, 1, 1, 1, np.array([72, 72])),
    ],
)
def test_transconv_size(
    size, kernel, stride, padding, dilation, output_padding, expected
):
    result = utils.transconv_size_2d(
        size, kernel, stride, padding, dilation, output_padding
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "size,expected",
    [
        (2, np.array([1, 1])),
        ((2, 2, 2), np.array([1, 1])),
        (1, np.array([0, 0])),
        ((1, 1, 1), np.array([0, 0])),
        (4, np.array([1, 1])),
        ((1, 4, 3), np.array([1, 0])),
    ],
)
def test_specify_output_padding(size, expected):
    result = utils.calc_output_padding_2d(size)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "length,kernel_size,stride,padding,out_padding",
    [
        (10, 2, 2, 1, 0),
        (11, 2, 2, 1, 0),
        (10, 2, 1, 1, 0),
        (11, 2, 1, 1, 0),
        (10, 2, 2, 0, 0),
        (11, 2, 2, 0, 0),
        (10, 2, 1, 0, 0),
        (11, 2, 1, 0, 0),
        (10, 3, 2, 1, 0),
        (11, 3, 2, 1, 0),
        (10, 3, 1, 1, 0),
        (11, 3, 1, 1, 0),
        (10, 3, 2, 0, 0),
        (11, 3, 2, 0, 0),
        (10, 3, 1, 0, 0),
        (11, 3, 1, 0, 0),
        (10, 2, 2, 1, 1),
        (11, 2, 2, 1, 1),
        (10, 2, 2, 0, 1),
        (11, 2, 2, 0, 1),
        (10, 3, 2, 1, 1),
        (11, 3, 2, 1, 1),
        (10, 3, 2, 0, 1),
        (11, 3, 2, 0, 1),
    ],
)
def test_1d_lengths(length, kernel_size, stride, padding, out_padding):
    H = 10
    x = randn(3, H, length)
    conv = nn.Conv1d(H, H, kernel_size, stride, padding)
    x = conv(x)
    L1 = x.shape[-1]
    assert L1 == utils.conv1d_size(length, kernel_size, stride, padding)
    deconv = nn.ConvTranspose1d(H, H, kernel_size, stride, padding, out_padding)
    x = deconv(x)
    L2 = x.shape[-1]
    assert L2 == utils.transconv_size_1d(
        L1, kernel_size, stride, padding, out_padding
    )


@pytest.mark.parametrize(
    "length,kernel_size,stride,padding",
    [
        (10, 2, 2, 1),
        (11, 2, 2, 1),
        (10, 2, 1, 1),
        (11, 2, 1, 1),
        (10, 2, 2, 0),
        (11, 2, 2, 0),
        (10, 2, 1, 0),
        (11, 2, 1, 0),
        (10, 3, 2, 1),
        (11, 3, 2, 1),
        (10, 3, 1, 1),
        (11, 3, 1, 1),
        (10, 3, 2, 0),
        (11, 3, 2, 0),
        (10, 3, 1, 0),
        (11, 3, 1, 0),
        (10, 2, 2, 1),
        (11, 2, 2, 1),
        (10, 2, 2, 0),
        (11, 2, 2, 0),
        (10, 3, 2, 1),
        (11, 3, 2, 1),
        (10, 3, 2, 0),
        (11, 3, 2, 0),
    ],
)
def test_find_out_padding(length, kernel_size, stride, padding):
    H = 10
    x = randn(3, H, length)
    conv = nn.Conv1d(H, H, kernel_size, stride, padding)
    x = conv(x)
    L1 = x.shape[-1]
    in_padding, out_padding = utils.calc_output_padding_1d(
        L1, length, kernel_size, stride, padding
    )
    deconv = nn.ConvTranspose1d(
        H, H, kernel_size, stride, in_padding, out_padding
    )
    x = deconv(x)
    L2 = x.shape[-1]
    assert L2 == length


@pytest.mark.parametrize(
    "data,target",
    [
        (tensor([[0, 2, 1, 1]]), tensor([[False, False, True, True]])),
        (tensor([[0, 2, 1, 2]]), tensor([[False, False, True, True]])),
        (
            tensor([[0, 2, 1, 1], [0, 2, 1, 2]]),
            tensor([[False, False, True, True], [False, False, True, True]]),
        ),
        (tensor([[0, 2, 3, 3]]), tensor([[False, False, False, False]])),
        (
            tensor([[0, 2, 1, 3], [0, 2, 3, 3]]),
            tensor([[False, False, True, True], [False, False, False, False]]),
        ),
        (tensor([[1, 1, 1, 1]]), tensor([[True, True, True, True]])),
    ],
    ids=["caseA", "caseB", "2Dcase", "missing_eos", "2Dmissing_eos", "all_eos"],
)
def test_mask_after_eos(data, target):
    eos = 1
    result = utils.mask_after_eos(data, eos)
    assert equal(result, target), f"Expected {target}, got {result}"


@pytest.mark.parametrize(
    "data,target",
    [
        (
            tensor([[[1, 0, 0, 1], [0, 0, 1, 2], [0, 1, 0, 1], [0, 1, 0, 1]]]),
            tensor(
                [[[1, 0, 0, 0.5], [0, 0, 1, 1], [0, 1, 0, 0.5], [0, 1, 0, 0.5]]]
            ),
        ),
        (  # N = 2
            tensor(
                [
                    [[1, 0, 0, 1], [0, 0, 1, 2], [0, 1, 0, 1], [0, 1, 0, 1]],
                    [[1, 0, 0, 1], [0, 0, 1, 2], [0, 1, 0, 1], [0, 1, 0, 1]],
                ]
            ),
            tensor(
                [
                    [
                        [1, 0, 0, 0.5],
                        [0, 0, 1, 1],
                        [0, 1, 0, 0.5],
                        [0, 1, 0, 0.5],
                    ],
                    [
                        [1, 0, 0, 0.5],
                        [0, 0, 1, 1],
                        [0, 1, 0, 0.5],
                        [0, 1, 0, 0.5],
                    ],
                ]
            ),
        ),
        (
            tensor([[[1, 0, 0, 1], [0, 0, 1, 2], [0, 1, 0, 1], [0, 0, 1, 1]]]),
            tensor(
                [[[1, 0, 0, 0.5], [0, 0, 1, 1], [0, 1, 0, 0.5], [0, 0, 1, 0.5]]]
            ),
        ),
        (  # no activities
            tensor([[[1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]]),
            tensor([[[1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]]),
        ),
        (  # no eos
            tensor([[[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]]),
            tensor([[[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]]),
        ),
        (  # no sos
            tensor([[[0, 0, 1, 1], [0, 0, 1, 1]]]),
            tensor([[[0, 0, 1, 1], [0, 0, 1, 1]]]),
        ),
        (  # 2 sos
            tensor([[[1, 0, 0, 1], [1, 0, 1, 1], [0, 0, 1, 2]]]),
            tensor([[[1, 0, 0, 1], [1, 0, 1, 1], [0, 0, 1, 2]]]),
        ),
        (  # sos not at start
            tensor([[[0, 0, 1, 1], [1, 0, 1, 1], [0, 0, 1, 1]]]),
            tensor([[[0, 0, 1, 1], [1, 0, 1, 1], [0, 0, 1, 1]]]),
        ),
        (  # sos after eos
            tensor([[[0, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 1]]]),
            tensor([[[0, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 1]]]),
        ),
    ],
    ids=[
        "caseA",
        "two_sequences",
        "caseB",
        "no_activities",
        "no_eos_in_sequence",
        "no_sos_in_sequence",
        "two_sos_in_sequence",
        "sos_not_at_start",
        "sos_after_eos",
    ],
)
def test_normalise_durations(data, target):
    result = utils.normalise_durations(data, sos=0, eos=1)
    assert result.shape == target.shape
    assert equal(result, target), f"Expected {target}, got {result}"


@pytest.mark.parametrize(
    "data,target",
    [
        (tensor([[1.0, 2.0, 3.0, 0.0]]), tensor([[0.0, 2.0, 0.0, 0.0]])),
        (tensor([[1.0, 2.0, 0.0, 0.0]]), tensor([[0.0, 0.0, 0.0, 0.0]])),
        (
            tensor([[1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 0.0, 0.0]]),
            tensor([[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        ),
        (tensor([[1.0, 2.0, 3.0, 4.0]]), tensor([[0.0, 2.0, 3.0, 0.0]])),
    ],
    ids=["caseA", "caseB", "combined_cases", "none"],
)
def test_durations_mask(data, target):
    duration_weights = utils.duration_mask(data)
    assert equal(
        duration_weights, target
    ), f"Expected {target}, got {duration_weights}"
