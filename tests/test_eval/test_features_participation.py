from numpy import array
from pandas import DataFrame

from caveat.evaluate.features import participation
from caveat.evaluate.features.utils import equals


def test_participation_prob_by_act():
    population = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "work"},
        ]
    )
    expected = {
        "home": (array([0, 1]), array([0, 2])),
        "work": (array([0, 1]), array([0, 2])),
    }
    result = participation.participation_prob_by_act(population)
    assert equals(result, expected)


def test_participation_rates():
    population = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "work"},
        ]
    )
    expected = {"all": (array([2, 3]), array([1, 1]))}
    result = participation.participation_rates(population)
    print(result)
    assert equals(result, expected)


def test_participation_rates_by_act():
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
        "home": (array([1, 2]), array([1, 1])),
        "work": (array([1]), array([2])),
    }
    assert equals(
        participation.participation_rates_by_act(population), expected
    )


def test_act_plan_seq_participation_rates():
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
        "0home": (array([1]), array([2])),
        "1work": (array([1]), array([2])),
        "2home": (array([0, 1]), array([1, 1])),
    }
    assert equals(
        participation.participation_rates_by_seq_act(population), expected
    )


def test_act_seq_participation_rates():
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
        "home0": (array([1]), array([2])),
        "work0": (array([1]), array([2])),
        "home1": (array([0, 1]), array([1, 1])),
    }
    assert equals(
        participation.participation_rates_by_act_enum(population), expected
    )


def test_combinations_with_replacement():
    array = ["a", "b", "c"]
    tuple_length = 2
    expected = [
        ["a", "a"],
        ["a", "b"],
        ["a", "c"],
        ["b", "b"],
        ["b", "c"],
        ["c", "c"],
    ]
    result = participation.combinations_with_replacement(array, tuple_length)
    assert result == expected


def test_calc_pair_rate():
    act_counts = DataFrame(
        {
            "home": [1, 2, 2, 2, 3],
            "work": [1, 1, 1, 1, 0],
            "school": [1, 0, 0, 0, 0],
        }
    )
    pair_rate = participation.calc_pair_rate(act_counts, ("home", "home"))
    assert pair_rate == {0: 1, 1: 4}


def test_participation_pairs():
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
        "home+work": (array([0, 1]), array([0, 2])),
        "home+home": (array([0, 1]), array([1, 1])),
        "work+work": (array([0, 1]), array([2, 0])),
    }
    assert equals(participation.joint_participation_prob(population), expected)
