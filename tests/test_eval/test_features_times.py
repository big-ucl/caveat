from numpy import array
from pandas import DataFrame

from caveat.evaluate.features import times
from caveat.evaluate.features.utils import equals


def test_times_by_act():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 2, "duration": 2},
            {"pid": 0, "act": "work", "start": 2, "end": 6, "duration": 4},
            {"pid": 0, "act": "home", "start": 6, "end": 8, "duration": 2},
            {"pid": 1, "act": "home", "start": 0, "end": 1, "duration": 1},
            {"pid": 1, "act": "work", "start": 1, "end": 8, "duration": 7},
        ]
    )
    expected_starts = {
        "home": (array([0, 6]) / 1440, array([2, 1])),
        "work": (array([1, 2]) / 1440, array([1, 1])),
    }
    expected_ends = {
        "home": (array([1, 2, 8]) / 1440, array([1, 1, 1])),
        "work": (array([6, 8]) / 1440, array([1, 1])),
    }
    expected_durations = {
        "home": (array([1, 2]) / 1440, array([1, 2])),
        "work": (array([4, 7]) / 1440, array([1, 1])),
    }
    assert equals(times.start_times_by_act(population), expected_starts)
    assert equals(times.end_times_by_act(population), expected_ends)
    assert equals(times.durations_by_act(population), expected_durations)


def test_start_and_duration_by_act_bins():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 2, "duration": 2},
            {"pid": 0, "act": "work", "start": 2, "end": 6, "duration": 4},
            {"pid": 0, "act": "home", "start": 6, "end": 8, "duration": 2},
            {"pid": 1, "act": "home", "start": 0, "end": 1, "duration": 1},
            {"pid": 1, "act": "work", "start": 1, "end": 8, "duration": 7},
        ]
    )
    expected = {
        "home": (
            array([[0.5, 1.5], [0.5, 2.5], [6.5, 2.5]]) / 1440,
            array([1, 1, 1]),
        ),
        "work": (array([[1.5, 7.5], [2.5, 4.5]]) / 1440, array([1, 1])),
    }
    assert equals(times.start_and_duration_by_act_bins(population, 1), expected)
    expected = {
        "home": (array([[2, 2], [6, 2]]) / 1440, array([2, 1])),
        "work": (array([[2, 6]]) / 1440, array([2])),
    }
    assert equals(times.start_and_duration_by_act_bins(population, 4), expected)


def test_start_times_by_act_plan_seq():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 2},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    expected = {
        "0home": (array([0]) / 1440, array([2])),
        "1work": (array([1, 2]) / 1440, array([1, 1])),
        "2home": (array([6]) / 1440, array([1])),
    }
    assert equals(times.start_times_by_act_plan_seq(population), expected)


def test_start_times_by_act_plan_enum():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 2},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    expected = {
        "home0": (array([0]) / 1440, array([2])),
        "work0": (array([1, 2]) / 1440, array([1, 1])),
        "home1": (array([6]) / 1440, array([1])),
    }
    assert equals(times.start_times_by_act_plan_enum(population), expected)


def test_end_times_by_act_plan_seq():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "end": 0},
            {"pid": 0, "act": "work", "end": 2},
            {"pid": 0, "act": "home", "end": 6},
            {"pid": 1, "act": "home", "end": 0},
            {"pid": 1, "act": "work", "end": 1},
        ]
    )
    expected = {
        "0home": (array([0]) / 1440, array([2])),
        "1work": (array([1, 2]) / 1440, array([1, 1])),
        "2home": (array([6]) / 1440, array([1])),
    }
    assert equals(times.end_times_by_act_plan_seq(population), expected)


def test_end_times_by_act_plan_enum():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "end": 0},
            {"pid": 0, "act": "work", "end": 2},
            {"pid": 0, "act": "home", "end": 6},
            {"pid": 1, "act": "home", "end": 0},
            {"pid": 1, "act": "work", "end": 1},
        ]
    )
    expected = {
        "home0": (array([0]) / 1440, array([2])),
        "work0": (array([1, 2]) / 1440, array([1, 1])),
        "home1": (array([6]) / 1440, array([1])),
    }
    assert equals(times.end_times_by_act_plan_enum(population), expected)


def test_durations_by_act_plan_seq():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 0},
            {"pid": 0, "act": "work", "duration": 2},
            {"pid": 0, "act": "home", "duration": 6},
            {"pid": 1, "act": "home", "duration": 0},
            {"pid": 1, "act": "work", "duration": 1},
        ]
    )
    expected = {
        "0home": (array([0]) / 1440, array([2])),
        "1work": (array([1, 2]) / 1440, array([1, 1])),
        "2home": (array([6]) / 1440, array([1])),
    }
    assert equals(times.durations_by_act_plan_seq(population), expected)


def test_durations_by_act_plan_enum():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 0},
            {"pid": 0, "act": "work", "duration": 2},
            {"pid": 0, "act": "home", "duration": 6},
            {"pid": 1, "act": "home", "duration": 0},
            {"pid": 1, "act": "work", "duration": 1},
        ]
    )
    expected = {
        "home0": (array([0]) / 1440, array([2])),
        "work0": (array([1, 2]) / 1440, array([1, 1])),
        "home1": (array([6]) / 1440, array([1])),
    }
    assert equals(times.durations_by_act_plan_enum(population), expected)
