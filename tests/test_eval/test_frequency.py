from numpy import array, array_equal
from pandas import DataFrame

from caveat.evaluate.features.frequency import (
    activity_densities,
    activity_frequencies,
    binned_activity_count,
    binned_activity_density,
)
from caveat.evaluate.features.utils import equals


def test_activity_count_bins():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 1, "duration": 1},
            {"pid": 0, "act": "work", "start": 1, "end": 2, "duration": 1},
            {"pid": 0, "act": "home", "start": 2, "end": 3, "duration": 1},
            {"pid": 1, "act": "home", "start": 0, "end": 1, "duration": 1},
            {"pid": 1, "act": "work", "start": 1, "end": 3, "duration": 2},
        ]
    )
    class_map = {"home": 0, "work": 1}
    binned = binned_activity_count(
        population, class_map=class_map, duration=3, step=1
    )
    expected = array([[2, 0], [0, 2], [1, 1]])
    assert array_equal(binned, expected)


def test_activity_density_bins():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 1, "duration": 1},
            {"pid": 0, "act": "work", "start": 1, "end": 2, "duration": 1},
            {"pid": 0, "act": "home", "start": 2, "end": 3, "duration": 1},
            {"pid": 1, "act": "home", "start": 0, "end": 1, "duration": 1},
            {"pid": 1, "act": "work", "start": 1, "end": 3, "duration": 2},
        ]
    )
    class_map = {"home": 0, "work": 1}
    binned = binned_activity_density(
        population, class_map=class_map, duration=3, step=1
    )
    expected = array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    assert array_equal(binned, expected)


def test_activity_frequencies():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 1, "duration": 1},
            {"pid": 0, "act": "work", "start": 1, "end": 2, "duration": 1},
            {"pid": 0, "act": "home", "start": 2, "end": 3, "duration": 1},
            {"pid": 1, "act": "home", "start": 0, "end": 1, "duration": 1},
            {"pid": 1, "act": "work", "start": 1, "end": 3, "duration": 2},
        ]
    )
    binned = activity_frequencies(population, 3, 1)
    print(binned)
    expected = {
        "home": (array([0, 1, 2]) / 3, array([2, 0, 1])),
        "work": (array([0, 1, 2]) / 3, array([0, 2, 1])),
    }
    assert equals(binned, expected)


def test_activity_frequencies_single_act():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 3, "duration": 3},
            {"pid": 1, "act": "home", "start": 0, "end": 3, "duration": 3},
        ]
    )
    binned = activity_frequencies(population, 3, 1)
    expected = {"home": (array([0, 1, 2]) / 3, array([2, 2, 2]))}
    assert equals(binned, expected)


def test_activity_densities():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 1, "duration": 1},
            {"pid": 0, "act": "work", "start": 1, "end": 2, "duration": 1},
            {"pid": 0, "act": "home", "start": 2, "end": 3, "duration": 1},
            {"pid": 1, "act": "home", "start": 0, "end": 1, "duration": 1},
            {"pid": 1, "act": "work", "start": 1, "end": 3, "duration": 2},
        ]
    )
    binned = activity_densities(population, 3, 1)
    expected = {
        "home": (array([0, 1, 2]) / 3, array([1, 0, 0.5])),
        "work": (array([0, 1, 2]) / 3, array([0, 1, 0.5])),
    }
    assert equals(binned, expected)


def test_activity_densities_single_act():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 3, "duration": 3},
            {"pid": 1, "act": "home", "start": 0, "end": 3, "duration": 3},
        ]
    )
    binned = activity_densities(population, 3, 1)
    expected = {"home": (array([0, 1, 2]) / 3, array([1, 1, 1]))}
    assert equals(binned, expected)
