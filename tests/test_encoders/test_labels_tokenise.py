import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from torch import Tensor
from torch.testing import assert_close

from caveat.label_encoding.tokenise import TokenAttributeEncoder


def test_encoder_ordinal():
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [10, 50, 20], "gender": ["M", "F", "F"]}
    )
    encoder = TokenAttributeEncoder(config={"age": "ordinal"})
    with pytest.raises(UserWarning) as w:
        _ = encoder.encode(data)
        assert w.message.contains("Ordinal encoding not supported")


def test_encoder_nominal():
    config = {"gender": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = TokenAttributeEncoder(config=config)
    encoded, weights = encoder.encode(data)
    expected = Tensor([[1], [0], [0]]).float()
    assert_close(encoded, expected)
    assert_close(weights, Tensor([[3], [3 / 2], [3 / 2]]).float())
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "type": "object",
    }


def test_re_encoder_nominal():
    config = {"gender": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = TokenAttributeEncoder(config=config)
    _ = encoder.encode(data)
    new = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "M", "F"]}
    )
    new_encoded, weights = encoder.encode(new)
    assert_close(new_encoded, Tensor([[1], [1], [0]]).float())
    assert_close(weights, Tensor([[3 / 2], [3 / 2], [3]]).float())
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "type": "object",
    }


def test_re_encoder_new_cat_nominal():
    config = {"gender": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = TokenAttributeEncoder(config=config)
    _ = encoder.encode(data)
    new = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "X", "F"]}
    )
    with pytest.raises(UserWarning) as w:
        encoder.encode(new)
        assert w.message.contains("X")


def test_encoder_missing_columns():
    config = {"unknown": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = TokenAttributeEncoder(config=config)
    with pytest.raises(UserWarning):
        encoder.encode(data)


def test_encoder_bad_config():
    bad_config = {"age": {"nominal": (0, 10)}}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = TokenAttributeEncoder(config=bad_config)
    with pytest.raises(UserWarning):
        encoder.encode(data)


def test_encoder_multi():
    config = {"gender": "nominal", "age": "nominal"}
    data = pd.DataFrame(
        {
            "pid": [0, 1, 2],
            "age": ["old", "old", "young"],
            "gender": ["M", "F", "F"],
        }
    )
    encoder = TokenAttributeEncoder(config=config)
    encoded, weights = encoder.encode(data)
    expected = Tensor([[1, 0], [0, 0], [0, 1]]).float()
    assert_close(encoded, expected)
    assert_close(
        weights, Tensor([[3, 3 / 2], [3 / 2, 3 / 2], [3 / 2, 3]]).float()
    )
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "type": "object",
    }
    assert encoder.config["age"] == {
        "nominal": {"old": 0, "young": 1},
        "location": 1,
        "type": "object",
    }
    assert encoder.label_kwargs == {"label_embed_sizes": [2, 2]}


def test_re_encoder_mixed():
    config = {"gender": "nominal", "age": "nominal"}
    data = pd.DataFrame(
        {
            "pid": [0, 1, 2],
            "age": ["old", "old", "young"],
            "gender": ["M", "F", "F"],
        }
    )
    encoder = TokenAttributeEncoder(config=config)
    _ = encoder.encode(data)
    new_data = pd.DataFrame(
        {
            "pid": [0, 1, 2],
            "age": ["young", "old", "young"],
            "gender": ["M", "M", "F"],
        }
    )
    new_encoded, _ = encoder.encode(new_data)
    expected = Tensor([[1, 1], [1, 0], [0, 1]]).float()
    assert_close(new_encoded, expected)
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "type": "object",
    }
    assert encoder.config["age"] == {
        "nominal": {"old": 0, "young": 1},
        "location": 1,
        "type": "object",
    }
    assert encoder.label_kwargs == {"label_embed_sizes": [2, 2]}


def test_decode_attributes():
    config = {"gender": "nominal", "age": "nominal"}
    data = pd.DataFrame(
        {
            "pid": [0, 1, 2],
            "age": ["old", "old", "young"],
            "gender": ["M", "F", "F"],
        }
    )
    encoder = TokenAttributeEncoder(config=config)
    encoded, _ = encoder.encode(data)
    expected = Tensor([[1, 0], [0, 0], [0, 1]]).float()
    assert_close(encoded, expected)

    preds = Tensor([[1, 0], [0, 0], [0, 1]])
    decoded_data = encoder.decode(preds)
    decoded_data = decoded_data[data.columns]
    assert_frame_equal(data, decoded_data, check_dtype=True, check_exact=True)

    preds = [
        Tensor([[0.1, 0.5], [0.9, 0.5], [0.5, 0.2]]),
        Tensor([[0.9, 0.3], [0.9, 0.3], [0.2, 0.5]]),
    ]
    decoded_data = encoder.argmax_decode(preds)[data.columns]
    assert_frame_equal(data, decoded_data, check_dtype=True, check_exact=True)
