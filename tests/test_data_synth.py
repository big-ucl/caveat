"""Tests for `caveat` package."""
import numpy as np
import pytest

from caveat.data import synth


@pytest.fixture
def single_state_transitions():
    return np.array([[[1]]])


@pytest.mark.parametrize(
    "a, b ,expected",
    [
        ((0, 0), (2, 0), (0, 0)),
        ((0, 0), (2, 2), (0, 1)),
        ((0, 0), (2, 3), (0, 1.5)),
        ((0, 2), (2, 1), (2, 1.5)),
        ((0, 1), (2, 0), (1, 0.5)),
        ((2, 2), (4, 1), (2, 1.5)),
        ((0, 0), (4, 2), (0, 0.5, 1, 1.5)),
    ],
)
def test_interpolate_pivot(a, b, expected):
    out = synth.interpolate_pivot(a, b)
    np.testing.assert_array_equal(out, np.array(expected))


@pytest.mark.parametrize(
    "pivots, size, expected",
    [
        ([(0, 1), (2, 1)], 2, [1, 1]),
        ([(0, 0), (2, 1)], 2, [0, 0.5]),
        ([(0, 1), (2, 0), (4, 1)], 4, [1, 0.5, 0, 0.5]),
    ],
)
def test_interpolate_from_pivots(pivots, size, expected):
    a = synth.interpolate_from_pivots(pivots, size, pivot_adjustment=1)
    np.testing.assert_array_equal(a, np.array(expected))


@pytest.mark.parametrize(
    "pivots, size, expected",
    [
        ([(0, 1), (2, 1)], 4, [1, 1, 1, 1]),
        ([(0, 0), (2, 1)], 4, [0, 0.25, 0.5, 0.75]),
        ([(0, 1), (2, 0), (4, 1)], 8, [1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75]),
    ],
)
def test_interpolate_from_pivots_with_adjustment(pivots, size, expected):
    a = synth.interpolate_from_pivots(pivots, size, pivot_adjustment=2)
    np.testing.assert_array_equal(a, np.array(expected))


def test_build_transitions_single_state():
    class SingleStateGen(synth.ActivityGen):
        duration = 20
        step_size = 5
        possible_states = ["home"]
        pivot_adjustment = 1
        transition_config = {"home": {"home": [(0, 1), (20, 1)]}}
        repetition_tollerance = np.array([1])
        repetition_sensitivity = np.array([1])
        min_duration_tollerance = np.array([1])
        min_duration_sensitivity = np.array([1])
        max_duration_tollerance = np.array([1])
        max_duration_sensitivity = np.array([1])

    gen = SingleStateGen()
    assert gen.steps == 4
    gen.build()
    np.testing.assert_array_equal(
        gen.transition_weights, np.array([[[1], [1], [1], [1]]])
    )


def test_build_transitions_single_state_with_adjusted_pivots():
    class SingleStateGen(synth.ActivityGen):
        duration = 20
        step_size = 5
        possible_states = ["home"]
        pivot_adjustment = 20
        transition_config = {"home": {"home": [(0, 1), (1, 1)]}}
        repetition_tollerance = np.array([1])
        repetition_sensitivity = np.array([1])
        min_duration_tollerance = np.array([1])
        min_duration_sensitivity = np.array([1])
        max_duration_tollerance = np.array([1])
        max_duration_sensitivity = np.array([1])

    gen = SingleStateGen()
    assert gen.steps == 4
    gen.build()
    np.testing.assert_array_equal(
        gen.transition_weights, np.array([[[1], [1], [1], [1]]])
    )


@pytest.fixture
def two_step_switch():
    class SingleStateGen(synth.ActivityGen):
        duration = 10
        step_size = 5
        possible_states = ["home", "work"]
        pivot_adjustment = 10
        transition_config = {
            "home": {"home": [(0, 0), (1, 0)], "work": [(0, 1), (1, 1)]},
            "work": {"home": [(0, 1), (1, 1)], "work": [(0, 0), (1, 0)]},
        }
        repetition_tollerance = np.array([1, 1])
        repetition_sensitivity = np.array([1, 1])
        min_duration_tollerance = np.array([1, 1])
        min_duration_sensitivity = np.array([1, 1])
        max_duration_tollerance = np.array([1, 1])
        max_duration_sensitivity = np.array([1, 1])

    gen = SingleStateGen()
    return gen


@pytest.fixture
def four_step_switch():
    class SingleStateGen(synth.ActivityGen):
        duration = 20
        step_size = 5
        possible_states = ["home", "work"]
        pivot_adjustment = 10
        transition_config = {
            "home": {"home": [(0, 0), (2, 0)], "work": [(0, 1), (2, 1)]},
            "work": {"home": [(0, 1), (2, 1)], "work": [(0, 0), (2, 0)]},
        }
        repetition_tollerance = np.array([1, 1])
        repetition_sensitivity = np.array([1, 1])
        min_duration_tollerance = np.array([1, 1])
        min_duration_sensitivity = np.array([1, 1])
        max_duration_tollerance = np.array([1, 1])
        max_duration_sensitivity = np.array([1, 1])

    gen = SingleStateGen()
    return gen


def test_build_transitions_simple_states(two_step_switch):
    assert two_step_switch.steps == 2
    two_step_switch.build()
    np.testing.assert_array_equal(
        two_step_switch.transition_weights,
        np.array([[[0, 1], [0, 1]], [[1, 0], [1, 0]]]),
    )


@pytest.fixture
def two_state_gen():
    class SingleStateGen(synth.ActivityGen):
        duration = 10
        step_size = 5
        possible_states = ["home", "work"]
        pivot_adjustment = 1
        transition_config = {
            "home": {"home": [(0, 1), (10, 0)], "work": [(0, 0), (10, 1)]},
            "work": {"home": [(0, 0), (10, 1)], "work": [(0, 1), (10, 0)]},
        }
        repetition_tollerance = np.array([1, 1])
        repetition_sensitivity = np.array([1, 1])
        min_duration_tollerance = np.array([1, 1])
        min_duration_sensitivity = np.array([1, 1])
        max_duration_tollerance = np.array([1, 1])
        max_duration_sensitivity = np.array([1, 1])

    gen = SingleStateGen()
    return gen


def test_build_transitions_states(two_state_gen):
    assert two_state_gen.steps == 2
    two_state_gen.build()
    np.testing.assert_array_equal(
        two_state_gen.transition_weights,
        np.array([[[1, 0], [0.5, 0.5]], [[0, 1], [0.5, 0.5]]]),
    )


def test_run_two_step_switch(two_step_switch):
    two_step_switch.build()
    trace = two_step_switch.run()
    assert trace == [(0, 0, 5, 5), (1, 5, 10, 5)]


def test_run_four_step_switch(four_step_switch):
    four_step_switch.build()
    trace = four_step_switch.run()
    assert trace == [
        (0, 0, 5, 5),
        (1, 5, 10, 5),
        (0, 10, 15, 5),
        (1, 15, 20, 5),
    ]
