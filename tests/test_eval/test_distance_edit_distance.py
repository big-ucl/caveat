import pytest
from torch import equal, tensor
from torchmetrics.text import EditDistance


@pytest.mark.parametrize(
    "target,preds,expected",
    [
        (
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor([[[1, 0, 2, 2]], [[1, 0, 2, 2]]]),
            tensor(0.0),
        )
    ],
)
def test_edit_distances(target, preds, expected):
    target = target.permute(0, 2, 1).squeeze()  # [B, T, C]
    preds = preds.permute(0, 2, 1).squeeze()  # [B, T, C]

    def to_string(xs):
        return " ".join([str(x) for x in xs])

    target = [to_string(x) for x in target.tolist()]
    preds = [to_string(x) for x in preds.tolist()]
    metric = EditDistance()
    assert equal(metric(preds, target), expected)
