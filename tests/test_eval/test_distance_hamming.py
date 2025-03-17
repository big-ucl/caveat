import pytest
from torch import equal, tensor
from torchmetrics.classification import MulticlassHammingDistance


@pytest.mark.parametrize(
    "target,preds,average,expected",
    [
        (
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            "none",
            tensor([0.0, 0.0, 0.0]),
        ),
        (
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            "macro",
            tensor(0.0),
        ),
        (
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            "micro",
            tensor(0.0),
        ),
        (
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor([[[1, 0, 2, 1]], [[1, 0, 2, 2]]]),
            "none",
            tensor([0.0, 0.0, 0.25]),
        ),
        (
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor([[[1, 0, 2, 1]], [[1, 0, 2, 2]]]),
            "macro",
            tensor(0.25 / 3),
        ),
        (
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor([[[1, 0, 2, 1]], [[1, 0, 2, 2]]]),
            "micro",
            tensor(0.125),
        ),
        (
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor([[[1, 0, 2, 1]], [[1, 0, 2, 2]]]),
            "weighted",
            tensor(0.125),
        ),
        (
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor([[[1, 1, 2, 1]], [[1, 0, 2, 2]]]),
            "none",
            tensor([0.5, 0.0, 0.25]),
        ),
        (
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor([[[1, 1, 2, 1]], [[1, 0, 2, 2]]]),
            "macro",
            tensor(0.25),
        ),
        (
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor([[[1, 1, 2, 1]], [[1, 0, 2, 2]]]),
            "micro",
            tensor(0.25),
        ),
        (
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor([[[1, 1, 2, 1]], [[1, 0, 2, 2]]]),
            "weighted",
            tensor(0.25),
        ),
    ],
)
def test_hamming_distance(target, preds, average, expected):
    target = target.permute(0, 2, 1)  # [B, T, C]
    preds = preds.permute(0, 2, 1)  # [B, T, C]
    metric = MulticlassHammingDistance(num_classes=3, average=average)
    assert equal(metric(preds, target), expected)
