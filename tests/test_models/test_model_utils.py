import numpy as np
import pytest
from torch import equal, tensor

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
    result = utils.transconv_size(
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
    result = utils.calc_output_padding(size)
    np.testing.assert_array_equal(result, expected)
