import pandas as pd
from torch import Tensor
from torch.testing import assert_close

from caveat.label_encoding.base import onehot_encode, ordinal_encode, tokenize


def test_tokenise_no_encodings():
    data = pd.Series(["M", "F", "F"])
    encoded, encodings = tokenize(data, None)
    assert_close(encoded, Tensor([1, 0, 0]).int())
    assert encodings == {"M": 1, "F": 0}


def test_tokenise_with_encodings():
    data = pd.Series(["M", "F", "F"])
    encodings = {"M": 0, "F": 1}
    encoded, encodings = tokenize(data, encodings)
    assert_close(encoded, Tensor([0, 1, 1]).int())
    assert encodings == {"M": 0, "F": 1}


def test_encode_ordinal():
    data = pd.Series([34, 96, 15])
    min, max = 0, 100
    encoded = ordinal_encode(data, min, max)
    expected = Tensor([[0.34], [0.96], [0.15]]).float()
    assert_close(encoded, expected)


def test_encode_nominal_no_encodings():
    data = pd.Series(["M", "F", "F"])
    encoded, encodings = onehot_encode(data, None)
    assert_close(encoded, Tensor([[0, 1], [1, 0], [1, 0]]).float())
    assert encodings == {"M": 1, "F": 0}


def test_encode_nominal_with_encodings():
    data = pd.Series(["M", "F", "F"])
    encodings = {"M": 0, "F": 1}
    encoded, encodings = onehot_encode(data, encodings)
    assert_close(encoded, Tensor([[1, 0], [0, 1], [0, 1]]).float())
    assert encodings == {"M": 0, "F": 1}
