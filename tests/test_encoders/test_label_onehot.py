import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from torch import Tensor, ones, ones_like
from torch.testing import assert_close

from caveat.label_encoding.onehot import OneHotAttributeEncoder


def test_encoder_ordinal():
    config = {"age": {"ordinal": [0, 100]}}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = OneHotAttributeEncoder(config=config)
    encoded, (weights, joint_weights) = encoder.encode(data)
    expected = Tensor([[0.34], [0.96], [0.15]]).float()
    assert_close(encoded, expected)
    assert_close(weights, ones_like(encoded))
    assert_close(joint_weights, ones(encoded.shape[0], 1))
    assert encoder.config["age"] == {
        "ordinal": [0, 100],
        "location": 0,
        "length": 1,
        "type": "int64",
    }


def test_re_encoder_ordinal():
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [10, 50, 20], "gender": ["M", "F", "F"]}
    )
    encoder = OneHotAttributeEncoder(config={"age": "ordinal"})
    encoded, (weights, joint_weights) = encoder.encode(data)
    expected = Tensor([[0.0], [1.0], [0.25]]).float()
    assert_close(encoded, expected)
    assert_close(weights, ones_like(encoded))
    assert_close(joint_weights, ones(encoded.shape[0], 1))
    new_data = pd.DataFrame({"pid": [0, 1, 2], "age": [20, 30, 40]})
    new_encoded, (weights, joint_weights) = encoder.encode(new_data)
    new_expected = Tensor([[0.25], [0.5], [0.75]]).float()
    assert_close(new_encoded, new_expected)
    assert_close(weights, ones_like(encoded))
    assert_close(joint_weights, ones(encoded.shape[0], 1))
    assert encoder.config["age"] == {
        "ordinal": [10, 50],
        "location": 0,
        "length": 1,
        "type": "int64",
    }


def test_re_encoder_ordinal_with_dtype_change():
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [10, 50, 20], "gender": ["M", "F", "F"]}
    )
    encoder = OneHotAttributeEncoder(config={"age": "ordinal"})
    encoded, _ = encoder.encode(data)
    expected = Tensor([[0.0], [1.0], [0.25]]).float()
    assert_close(encoded, expected)
    new_data = pd.DataFrame({"pid": [0, 1, 2], "age": [20.0, 30.0, 40.0]})
    with pytest.raises(UserWarning) as w:
        encoder.encode(new_data)
        assert w.message.contains("int64")
        assert w.message.contains("float64")


def test_encoder_nominal():
    config = {"gender": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = OneHotAttributeEncoder(config=config)
    encoded, _ = encoder.encode(data)
    expected = Tensor([[0, 1], [1, 0], [1, 0]]).float()
    assert_close(encoded, expected)
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "length": 2,
        "type": "object",
    }


def test_re_encoder_nominal():
    config = {"gender": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = OneHotAttributeEncoder(config=config)
    encoded, _ = encoder.encode(data)
    assert_close(encoded, Tensor([[0, 1], [1, 0], [1, 0]]).float())
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "length": 2,
        "type": "object",
    }
    new = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "M", "F"]}
    )
    new_encoded, _ = encoder.encode(new)
    assert_close(new_encoded, Tensor([[0, 1], [0, 1], [1, 0]]).float())
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "length": 2,
        "type": "object",
    }


def test_re_encoder_new_cat_nominal():
    config = {"gender": "nominal"}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = OneHotAttributeEncoder(config=config)
    encoded, _ = encoder.encode(data)
    assert_close(encoded, Tensor([[0, 1], [1, 0], [1, 0]]).float())
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "length": 2,
        "type": "object",
    }
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
    encoder = OneHotAttributeEncoder(config=config)
    with pytest.raises(UserWarning):
        encoder.encode(data)


def test_encoder_bad_config():
    bad_config = {"age": {"nominal": (0, 10)}}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = OneHotAttributeEncoder(config=bad_config)
    with pytest.raises(UserWarning):
        encoder.encode(data)


def test_encoder_mixed():
    config = {"gender": "nominal", "age": {"ordinal": [0, 100]}}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = OneHotAttributeEncoder(config=config)
    encoded, _ = encoder.encode(data)
    expected = Tensor([[0, 1, 0.34], [1, 0, 0.96], [1, 0, 0.15]]).float()
    assert_close(encoded, expected)
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "length": 2,
        "type": "object",
    }
    assert encoder.config["age"] == {
        "ordinal": [0, 100],
        "location": 2,
        "length": 1,
        "type": "int64",
    }


def test_re_encoder_mixed():
    config = {"gender": "nominal", "age": {"ordinal": [0, 100]}}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = OneHotAttributeEncoder(config=config)
    encoded, _ = encoder.encode(data)
    assert_close(
        encoded, Tensor([[0, 1, 0.34], [1, 0, 0.96], [1, 0, 0.15]]).float()
    )
    new_data = pd.DataFrame(
        {"pid": [3, 4, 5], "age": [25, 50, 75], "gender": ["F", "M", "F"]}
    )
    new_encoded, _ = encoder.encode(new_data)
    new_expected = Tensor([[1, 0, 0.25], [0, 1, 0.5], [1, 0, 0.75]]).float()
    assert_close(new_encoded, new_expected)
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "length": 2,
        "type": "object",
    }
    assert encoder.config["age"] == {
        "ordinal": [0, 100],
        "location": 2,
        "length": 1,
        "type": "int64",
    }


def test_re_encoder_mixed_re_ordered_dataframe():
    config = {"gender": "nominal", "age": {"ordinal": [0, 100]}}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    encoder = OneHotAttributeEncoder(config=config)
    encoded, _ = encoder.encode(data)
    assert_close(
        encoded, Tensor([[0, 1, 0.34], [1, 0, 0.96], [1, 0, 0.15]]).float()
    )
    new_data = pd.DataFrame(
        {"pid": [0, 1, 2], "gender": ["F", "M", "F"], "age": [25, 50, 75]}
    )
    new_encoded, _ = encoder.encode(new_data)
    new_expected = Tensor([[1, 0, 0.25], [0, 1, 0.5], [1, 0, 0.75]]).float()
    assert_close(new_encoded, new_expected)
    assert encoder.config["gender"] == {
        "nominal": {"M": 1, "F": 0},
        "location": 0,
        "length": 2,
        "type": "object",
    }
    assert encoder.config["age"] == {
        "ordinal": [0, 100],
        "location": 2,
        "length": 1,
        "type": "int64",
    }


def test_decode_attributes():
    config = {"gender": "nominal", "age": {"ordinal": [0, 100]}}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [30, 90, 25], "gender": ["M", "F", "F"]}
    )
    encoder = OneHotAttributeEncoder(config=config)
    encoded, _ = encoder.encode(data)
    assert_close(
        encoded, Tensor([[0, 1, 0.30], [1, 0, 0.90], [1, 0, 0.25]]).float()
    )
    decoded_data = encoder.decode(encoded)[data.columns]
    assert_frame_equal(data, decoded_data, check_dtype=True, check_exact=True)


def test_re_encoder_mixed_re_ordered_dataframe_and_decode():
    config = {"gender": "nominal", "age": {"ordinal": [0, 100]}}
    data = pd.DataFrame(
        {"pid": [0, 1, 2], "age": [34, 96, 15], "gender": ["M", "F", "F"]}
    )
    new_data = pd.DataFrame(
        {"pid": [0, 1, 2], "gender": ["F", "M", "F"], "age": [25, 50, 75]}
    )
    encoder = OneHotAttributeEncoder(config=config)
    encoded, _ = encoder.encode(data)
    new_encoded, _ = encoder.encode(new_data)

    decoded_data = encoder.decode(encoded)[data.columns]
    assert_frame_equal(data, decoded_data, check_dtype=True, check_exact=True)

    new_decoded_data = encoder.decode(new_encoded)[new_data.columns]
    assert_frame_equal(
        new_data, new_decoded_data, check_dtype=True, check_exact=True
    )
    decoded_data = encoder.decode(encoded)[data.columns]
    assert_frame_equal(data, decoded_data, check_dtype=True, check_exact=True)


# def test_load_and_encode_attributes(tmpdir):
#     config = {
#         "schedules_path": "tests/fixtures/schedules.csv",
#         "attributes_path": "tests/fixtures/attributes.csv",
#         "conditionals": {"gender": "nominal", "age": {"ordinal": [0, 100]}},
#         "encoder_params": {
#             "name": "discrete",
#             "duration": 1440,
#             "step_size": 20,
#         },
#     }
#     schedules, attributes, synthetic_attributes = load_data(config)
#     assert_frame_equal(attributes, synthetic_attributes)
#     attribute_encoder, encoded_attributes = encode_input_attributes(
#         tmpdir, attributes, config
#     )
#     schedule_encoder, encoded_schedules, data_loader = encode_schedules(
#         tmpdir, schedules, encoded_attributes, config
#     )

#     i = data_loader
#     print(i)
