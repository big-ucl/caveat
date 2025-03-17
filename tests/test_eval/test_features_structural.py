from numpy import array
from pandas import DataFrame, MultiIndex, Series, concat

from caveat.evaluate import evaluate
from caveat.evaluate.features.structural import (
    contains_consecutive,
    duration_consistency,
    start_and_end_acts,
    structural_eval,
    time_consistency,
)
from caveat.evaluate.features.utils import equals


def test_start_and_end_acts():
    population = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "work"},
        ]
    )
    expected = {
        "first act home": (array([0, 1]), array([0, 2])),
        "last act home": (array([0, 1]), array([1, 1])),
    }
    assert equals(start_and_end_acts(population, target="home"), expected)


def test_time_consistency():
    population = DataFrame(
        [
            {"pid": 0, "start": 0, "end": 10, "duration": 10},
            {"pid": 0, "start": 10, "end": 20, "duration": 10},
            {"pid": 0, "start": 20, "end": 30, "duration": 10},
            {"pid": 1, "start": 0, "end": 10, "duration": 10},
            {"pid": 1, "start": 10, "end": 20, "duration": 10},
        ]
    )
    expected = {
        "starts at 0": (array([0, 1]), array([0, 2])),
        "ends at 30": (array([0, 1]), array([1, 1])),
        "duration is 30": (array([0, 1]), array([1, 1])),
    }
    assert equals(time_consistency(population, target=30), expected)


def test_duration_consistency():
    population = DataFrame(
        [
            {"pid": 0, "start": 0, "end": 10, "duration": 10},
            {"pid": 0, "start": 10, "end": 20, "duration": 10},
            {"pid": 0, "start": 20, "end": 30, "duration": 10},
            {"pid": 1, "start": 0, "end": 10, "duration": 10},
            {"pid": 1, "start": 10, "end": 20, "duration": 10},
        ]
    )
    expected = {"total duration": (array([20, 30]), array([1, 1]))}
    assert equals(duration_consistency(population, factor=1), expected)


def test_does_not_contains_consecutive():
    schedule = DataFrame(
        [
            {"act": "home"},
            {"act": "work"},
            {"act": "home"},
            {"act": "work"},
            {"act": "home"},
        ]
    )
    assert not contains_consecutive(schedule, act="home")
    assert not contains_consecutive(schedule, act="work")


def test_contains_consecutive():
    schedule = DataFrame(
        [
            {"act": "home"},
            {"act": "home"},
            {"act": "work"},
            {"act": "home"},
            {"act": "work"},
        ]
    )
    assert contains_consecutive(schedule, act="home")
    assert not contains_consecutive(schedule, act="work")


def test_structural_eval():
    schedule = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 2, "act": "home"},
            {"pid": 2, "act": "work"},
            {"pid": 2, "act": "shop"},
        ]
    )
    weights, metrics = structural_eval(schedule, "observed")
    assert weights.reset_index(drop=True).equals(
        Series([3, 3, 3, 3, 3, 3, 3, 3])
    )
    print(metrics)
    assert metrics.reset_index(drop=True).equals(
        Series([2 / 3, 1 / 3, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0])
    )


def test_describe_structural():
    index = MultiIndex.from_tuples(
        [
            ("sample quality", "invalid", "all"),
            ("sample quality", "not home based", "all"),
            ("sample quality", "not home based", "starts"),
            ("sample quality", "not home based", "ends"),
            ("sample quality", "consecutive", "all"),
            ("sample quality", "consecutive", "home"),
            ("sample quality", "consecutive", "work"),
            ("sample quality", "consecutive", "education"),
        ],
        names=["domain", "feature", "segment"],
    )

    observed_weights = Series(
        [3, 3, 3, 3, 3, 3, 3, 3], index=index, name="observed__weight"
    )
    observed_metrics = Series(
        [2 / 3, 1 / 3, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0],
        index=index,
        name="observed",
    )
    weights = Series([3, 3, 3, 3, 3, 3, 3, 3], index=index, name="y__weight")
    metrics = Series(
        [2 / 3, 1 / 3, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0], index=index, name="y"
    )
    metrics = concat(
        [observed_weights, observed_metrics, weights, metrics], axis=1
    )
    metrics["unit"] = "prob. invalid"
    frames = evaluate.describe(metrics, metrics)
    assert len(frames["feature_descriptions"]) == 3
    assert len(frames["domain_descriptions"]) == 1
    assert len(frames["feature_distances"]) == 3
    assert len(frames["domain_distances"]) == 1


def test_describe_splits_structural():
    index = MultiIndex.from_tuples(
        [
            ("sample quality", "invalid", "all", "a"),
            ("sample quality", "not home based", "all", "a"),
            ("sample quality", "not home based", "starts", "a"),
            ("sample quality", "not home based", "ends", "a"),
            ("sample quality", "consecutive", "all", "a"),
            ("sample quality", "consecutive", "home", "a"),
            ("sample quality", "consecutive", "work", "a"),
            ("sample quality", "consecutive", "education", "a"),
            ("sample quality", "invalid", "all", "b"),
            ("sample quality", "not home based", "all", "b"),
            ("sample quality", "not home based", "starts", "b"),
            ("sample quality", "not home based", "ends", "b"),
            ("sample quality", "consecutive", "all", "b"),
            ("sample quality", "consecutive", "home", "b"),
            ("sample quality", "consecutive", "work", "b"),
            ("sample quality", "consecutive", "education", "b"),
        ],
        names=["domain", "feature", "segment", "sub_pop"],
    )

    observed_weights = Series([3] * 16, index=index, name="observed__weight")
    observed_metrics = Series([1 / 3] * 16, index=index, name="observed")
    weights = Series([3] * 16, index=index, name="y__weight")
    metrics = Series([1 / 3, 0] * 8, index=index, name="y")
    metrics = concat(
        [observed_weights, observed_metrics, weights, metrics], axis=1
    )
    metrics["unit"] = "prob. invalid"
    frames = evaluate.describe_splits(metrics, metrics)
    assert len(frames["feature_descriptions"]) == 6
    assert len(frames["domain_descriptions"]) == 1
    assert len(frames["feature_distances"]) == 6
    assert len(frames["domain_distances"]) == 1
