from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"
SCHEDULES_PATH = FIXTURES / "test_schedules.csv"
ATTRIBUTES_PATH = FIXTURES / "test_attributes.csv"


# ---------------------------------------------------------------------------
# evaluate_synthetics
# ---------------------------------------------------------------------------


@patch("caveat.runners.evaluate")
@patch("caveat.runners.data")
def test_evaluate_synthetics_basic(mock_data, mock_evaluate, tmp_path):
    """No split_on: calls evaluate.evaluate() and evaluate.report()."""
    from caveat.runners import evaluate_synthetics

    schedules = pd.read_csv(SCHEDULES_PATH)
    synthetic_schedules = {"model_a": schedules.copy()}
    synthetic_labels = {"model_a": None}
    eval_params = {}

    mock_evaluate.evaluate.return_value = {"model_a": MagicMock()}

    evaluate_synthetics(
        synthetic_schedules=synthetic_schedules,
        synthetic_labels=synthetic_labels,
        default_eval_schedules=schedules,
        default_eval_attributes=None,
        write_path=tmp_path,
        eval_params=eval_params,
    )

    mock_evaluate.evaluate.assert_called_once()
    mock_evaluate.report.assert_called_once()
    mock_evaluate.compare_splits.assert_not_called()
    mock_evaluate.report_splits.assert_not_called()


@patch("caveat.runners.evaluate")
@patch("caveat.runners.data")
def test_evaluate_synthetics_with_split_on(mock_data, mock_evaluate, tmp_path):
    """split_on set: calls compare_splits() and report_splits()."""
    from caveat.runners import evaluate_synthetics

    schedules = pd.read_csv(SCHEDULES_PATH)
    synthetic_schedules = {"model_a": schedules.copy()}
    synthetic_labels = {"model_a": None}
    eval_params = {"split_on": ["gender"]}

    mock_evaluate.compare_splits.return_value = {"gender": MagicMock()}
    mock_evaluate.evaluate.return_value = {"model_a": MagicMock()}

    evaluate_synthetics(
        synthetic_schedules=synthetic_schedules,
        synthetic_labels=synthetic_labels,
        default_eval_schedules=schedules,
        default_eval_attributes=None,
        write_path=tmp_path,
        eval_params=eval_params,
    )

    mock_evaluate.compare_splits.assert_called_once()
    call_kwargs = mock_evaluate.compare_splits.call_args.kwargs
    assert call_kwargs["observed"] is schedules
    assert call_kwargs["split_on"] == ["gender"]

    mock_evaluate.report_splits.assert_called_once()
    mock_evaluate.evaluate.assert_called_once()
    mock_evaluate.report.assert_called_once()


@patch("caveat.runners.evaluate")
@patch("caveat.runners.data")
def test_evaluate_synthetics_custom_schedules_path(mock_data, mock_evaluate, tmp_path):
    """eval_params has schedules_path: custom schedules are loaded instead of default."""
    from caveat.runners import evaluate_synthetics

    default_schedules = pd.read_csv(SCHEDULES_PATH)
    custom_schedules = default_schedules.copy()
    mock_data.load_and_validate_schedules.return_value = custom_schedules

    synthetic_schedules = {"model_a": default_schedules.copy()}
    synthetic_labels = {"model_a": None}
    eval_params = {"schedules_path": str(SCHEDULES_PATH)}

    mock_evaluate.evaluate.return_value = {"model_a": MagicMock()}

    evaluate_synthetics(
        synthetic_schedules=synthetic_schedules,
        synthetic_labels=synthetic_labels,
        default_eval_schedules=default_schedules,
        default_eval_attributes=None,
        write_path=tmp_path,
        eval_params=eval_params,
    )

    mock_data.load_and_validate_schedules.assert_called_once_with(str(SCHEDULES_PATH))

    call_kwargs = mock_evaluate.evaluate.call_args.kwargs
    assert call_kwargs["target_schedules"] is custom_schedules


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
    mock_data.load_and_validate_attributes.return_value = (attributes, attributes)

    config = {
        "schedules_path": str(SCHEDULES_PATH),
        "attributes_path": str(ATTRIBUTES_PATH),
    }
    result_schedules, attrs, synth_attrs = load_data(config)

    mock_data.load_and_validate_schedules.assert_called_once()
    mock_data.load_and_validate_attributes.assert_called_once()
    assert attrs is attributes
    assert synth_attrs is attributes
