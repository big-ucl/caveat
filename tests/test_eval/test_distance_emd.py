import pytest
from scipy.stats import wasserstein_distance
from torch import tensor

from caveat.evaluate.distance.wasserstein import SinkhornDistance


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (tensor([1.0, 1.0, 1.0]), tensor([0.0, 0.0, 0.0]), 1.0),
        (tensor([1.0, 1.0, 1.0, 1.0]), tensor([0.0, 0.0, 0.0, 1.0]), 0.75),
        (tensor([1.0, 1.0, 1.0]), tensor([0.0, 0.0]), 1.0),
    ],
)
def test_wasserstein_1d(x, y, expected):
    dist = wasserstein_distance(x, y)
    assert dist == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (
            tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
            tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            1.0,
        ),
        (
            tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
            tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]]),
            0.5,
        ),
    ],
)
def test_custom_sinkhorn_2d(x, y, expected):
    sinkhorn = SinkhornDistance(eps=0.01, max_iter=100, reduction=None)
    dist, P, C = sinkhorn(x, y)
    assert dist.item() == pytest.approx(expected, rel=1e-1 * 2)


# @pytest.mark.parametrize(
#     "x,y,expected",
#     [
#         (
#             tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
#             tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
#             1.0,
#         ),
#         (
#             tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
#             tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]]),
#             0.5,
#         ),
#     ],
# )
# def test_custom_wasserstein_2d_slicer(x, y, expected):
#     dist = sliced_wasserstein(x, y, 1000)
#     assert dist == pytest.approx(expected, rel=1e-1 * 2)


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (
            tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]),
            tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            1.0,
        ),
        (
            tensor(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                ]
            ),
            tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                ]
            ),
            0.5,
        ),
    ],
)
def test_custom_sinkhorn_3d(x, y, expected):
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    dist, P, C = sinkhorn(x, y)
    assert dist.item() == pytest.approx(expected, rel=1e-1 * 2)
