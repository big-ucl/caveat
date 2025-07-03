from numpy import array
from pandas import DataFrame, MultiIndex, Series, concat

from caveat.evaluate import evaluate
from caveat.evaluate.features.structural import (
    contains_consecutive,
    duration_consistency,
    feasibility_eval,
    start_and_end_acts,
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


def test_feasibility_eval():
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
    weights, metrics = feasibility_eval(schedule, "observed")
    assert (
        weights.reset_index(drop=True)
        .astype("int32")
        .equals(Series([3, 3, 3, 3, 3, 3, 3, 3], dtype="int32"))
    )
    assert metrics.reset_index(drop=True).equals(
        Series([2 / 3, 1 / 3, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0])
    )


def test_describe_structural():
    index = MultiIndex.from_tuples(
        [
            ("feasibility", "invalid", "all"),
            ("feasibility", "not home based", "all"),
            ("feasibility", "not home based", "starts"),
            ("feasibility", "not home based", "ends"),
            ("feasibility", "consecutive", "all"),
            ("feasibility", "consecutive", "home"),
            ("feasibility", "consecutive", "work"),
            ("feasibility", "consecutive", "education"),
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
    assert len(frames["descriptions"]) == 8
    assert len(frames["group_descriptions"]) == 3
    assert len(frames["domain_descriptions"]) == 1

    assert len(frames["distances"]) == 8
    assert len(frames["group_distances"]) == 3
    assert len(frames["domain_distances"]) == 1


def test_describe_splits_structural():
    index = MultiIndex.from_tuples(
        [
            ("feasibility", "invalid", "all", "a"),
            ("feasibility", "not home based", "all", "a"),
            ("feasibility", "not home based", "starts", "a"),
            ("feasibility", "not home based", "ends", "a"),
            ("feasibility", "consecutive", "all", "a"),
            ("feasibility", "consecutive", "home", "a"),
            ("feasibility", "consecutive", "work", "a"),
            ("feasibility", "consecutive", "education", "a"),
            ("feasibility", "invalid", "all", "b"),
            ("feasibility", "not home based", "all", "b"),
            ("feasibility", "not home based", "starts", "b"),
            ("feasibility", "not home based", "ends", "b"),
            ("feasibility", "consecutive", "all", "b"),
            ("feasibility", "consecutive", "home", "b"),
            ("feasibility", "consecutive", "work", "b"),
            ("feasibility", "consecutive", "education", "b"),
        ],
        names=["domain", "feature", "segment", "label"],
    )

    observed_weights = Series([3] * 16, index=index, name="observed__weight")
    observed_metrics = Series([1 / 3] * 16, index=index, name="observed")
    weights = Series([3] * 16, index=index, name="y__weight")
    metrics = Series([1 / 3, 0] * 8, index=index, name="y")
    metrics = concat(
        [observed_weights, observed_metrics, weights, metrics], axis=1
    )
    metrics["unit"] = "prob. invalid"
    frames = evaluate.describe(metrics, metrics)
    print(frames["descriptions"])
    assert len(frames["descriptions"]) == 8
    assert len(frames["group_descriptions"]) == 3
    assert len(frames["domain_descriptions"]) == 1

    assert len(frames["distances"]) == 8
    assert len(frames["group_distances"]) == 3
    assert len(frames["domain_distances"]) == 1

    frames = evaluate.describe_labels(metrics, metrics)
    assert len(frames["label_descriptions"]) == 16
    assert len(frames["label_group_descriptions"]) == 6
    assert len(frames["label_domain_descriptions"]) == 2

    assert len(frames["label_distances"]) == 16
    assert len(frames["label_group_distances"]) == 6
    assert len(frames["label_domain_distances"]) == 2
