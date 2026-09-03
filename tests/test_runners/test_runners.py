from pathlib import Path
from unittest.mock import patch

import pandas as pd

FIXTURES = Path(__file__).parent.parent / "fixtures"
SCHEDULES_PATH = FIXTURES / "test_schedules.csv"
ATTRIBUTES_PATH = FIXTURES / "test_attributes.csv"


# ---------------------------------------------------------------------------
# evaluate_synthetics
# ---------------------------------------------------------------------------


@patch("caveat.runners.Evaluator")
@patch("caveat.runners.data")
def test_evaluate_synthetics_basic(mock_data, mock_evaluator_cls, tmp_path):
    """No split_on: calls compare() and not compare_splits()."""
    from caveat.runners import evaluate_synthetics

    schedules = pd.read_csv(SCHEDULES_PATH)
    synthetic_schedules = {"model_a": schedules.copy()}
    synthetic_labels = {"model_a": None}
    eval_params = {}

    mock_evaluator = mock_evaluator_cls.return_value
    mock_result = mock_evaluator.compare.return_value

    evaluate_synthetics(
        synthetic_schedules=synthetic_schedules,
        synthetic_labels=synthetic_labels,
        default_eval_schedules=schedules,
        default_eval_attributes=None,
        write_path=tmp_path,
        eval_params=eval_params,
    )

    # Evaluator instantiated without split_on/target_attributes
    mock_evaluator_cls.assert_called_once_with(target=schedules, progress=False)

    # compare called without `attributes`
    mock_evaluator.compare.assert_called_once_with(
        synthetic=synthetic_schedules, verbose=False
    )

    mock_result.save.assert_called_once_with(tmp_path / "eval")


@patch("caveat.runners.Evaluator")
@patch("caveat.runners.data")
def test_evaluate_synthetics_with_split_on(
    mock_data, mock_eval_class, tmp_path
):
    """split_on set: Evaluator gets target_attributes/split_on,
    compare() gets attributes=synthetic_labels."""
    from caveat.runners import evaluate_synthetics

    schedules = pd.read_csv(SCHEDULES_PATH)
    synthetic_schedules = {"model_a": schedules.copy()}
    synthetic_labels = {"model_a": None}
    # split_on requires attributes to be available for eval_schedules
    eval_attributes = pd.DataFrame(
        {"pid": schedules["pid"].unique(), "gender": "f"}
    )
    eval_params = {"split_on": ["gender"]}

    mock_evaluator = mock_eval_class.return_value
    mock_result = mock_evaluator.compare.return_value
    mock_result.rank_models.return_value = {"model_a": 0.5}
    mock_result.best_model = "model_a"

    evaluate_synthetics(
        synthetic_schedules=synthetic_schedules,
        synthetic_labels=synthetic_labels,
        default_eval_schedules=schedules,
        default_eval_attributes=eval_attributes,
        write_path=tmp_path,
        eval_params=eval_params,
    )

    mock_eval_class.assert_called_once_with(
        target=schedules,
        target_attributes=eval_attributes,
        split_on=["gender"],
        progress=False,
    )

    mock_evaluator.compare.assert_called_once_with(
        synthetic=synthetic_schedules,
        attributes=synthetic_labels,
        verbose=False,
    )

    mock_result.save.assert_called_once_with(tmp_path / "eval")


@patch("caveat.runners.data")
@patch("caveat.runners.Evaluator")
def test_evaluate_synthetics_custom_schedules_path(
    mock_eval_class, mock_data, tmp_path
):
    """eval_params has schedules_path: custom schedules are loaded instead of default."""
    from caveat.runners import evaluate_synthetics

    default_schedules = pd.read_csv(SCHEDULES_PATH)
    custom_schedules = default_schedules.copy()
    mock_data.load_and_validate_schedules.return_value = custom_schedules

    synthetic_schedules = {"model_a": default_schedules.copy()}
    synthetic_labels = {"model_a": None}
    eval_params = {"schedules_path": str(SCHEDULES_PATH)}

    mock_evaluator = mock_eval_class.return_value
    mock_result = mock_evaluator.compare.return_value
    mock_result.rank_models.return_value = {"model_a": 0.5}
    mock_result.best_model = "model_a"

    evaluate_synthetics(
        synthetic_schedules=synthetic_schedules,
        synthetic_labels=synthetic_labels,
        default_eval_schedules=default_schedules,
        default_eval_attributes=None,
        write_path=tmp_path,
        eval_params=eval_params,
    )

    mock_data.load_and_validate_schedules.assert_called_once_with(
        str(SCHEDULES_PATH)
    )

    # the custom (loaded) schedules should flow into Evaluator as `target`,
    # not the default_schedules
    mock_eval_class.assert_called_once_with(
        target=custom_schedules, progress=False
    )

    mock_evaluator.compare.assert_called_once_with(
        synthetic=synthetic_schedules, verbose=False
    )


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


@patch("caveat.runners.data")
def test_load_data_no_attributes(mock_data):
    """Config without attributes_path returns schedules and None attributes."""
    from caveat.runners import load_data

    schedules = pd.read_csv(SCHEDULES_PATH)
    mock_data.load_and_validate_schedules.return_value = schedules
    mock_data.load_and_validate_attributes.return_value = (None, None)

    config = {"schedules_path": str(SCHEDULES_PATH)}
    result_schedules, attrs, synth_attrs = load_data(config)

    mock_data.load_and_validate_schedules.assert_called_once()
    assert attrs is None
    assert synth_attrs is None


@patch("caveat.runners.data")
def test_load_data_with_attributes(mock_data):
    """Config with attributes_path returns both schedules and attributes."""
    from caveat.runners import load_data

    schedules = pd.read_csv(SCHEDULES_PATH)
    attributes = pd.read_csv(ATTRIBUTES_PATH)
    mock_data.load_and_validate_schedules.return_value = schedules
    mock_data.load_and_validate_attributes.return_value = (
        attributes,
        attributes,
    )

    config = {
        "schedules_path": str(SCHEDULES_PATH),
        "attributes_path": str(ATTRIBUTES_PATH),
    }
    result_schedules, attrs, synth_attrs = load_data(config)

    mock_data.load_and_validate_schedules.assert_called_once()
    mock_data.load_and_validate_attributes.assert_called_once()
    assert attrs is attributes
    assert synth_attrs is attributes
