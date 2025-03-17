from numpy import array
from pandas import DataFrame

from caveat.evaluate.features.structural import (
    contains_consecutive,
    duration_consistency,
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
