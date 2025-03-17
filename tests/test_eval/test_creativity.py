from pandas import DataFrame

from caveat.evaluate.features.creativity import (
    conservatism,
    diversity,
    hash_population,
    hash_schedule,
    homogeneity,
    novelty,
)


def test_hash_schedule():
    schedule = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
        ]
    )
    assert hash_schedule(schedule) == "home10work10home10"


def test_hash_population():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    assert hash_population(population) == {"home10work10", "home10work10home10"}


def test_internal_uniqueness_full():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    hashed = hash_population(population)
    assert diversity(population, hashed) == 1
    assert homogeneity(population, hashed) == 0


def test_internal_uniqueness_half():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
        ]
    )
    hashed = hash_population(population)
    assert diversity(population, hashed) == 0.5
    assert homogeneity(population, hashed) == 0.5


def test_novelty_none():
    a = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    b = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    assert novelty(hash_population(a), hash_population(b)) == 0
    assert conservatism(hash_population(a), hash_population(b)) == 1


def test_novelty_full():
    a = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    b = DataFrame(
        [
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 2, "act": "work", "duration": 15},
            {"pid": 2, "act": "home", "duration": 5},
            {"pid": 3, "act": "home", "duration": 10},
            {"pid": 3, "act": "work", "duration": 10},
            {"pid": 3, "act": "shop", "duration": 10},
        ]
    )
    assert novelty(hash_population(a), hash_population(b)) == 1
    assert conservatism(hash_population(a), hash_population(b)) == 0


def test_novelty_partial():
    a = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    b = DataFrame(
        [
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 2, "act": "work", "duration": 10},
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 3, "act": "home", "duration": 10},
            {"pid": 3, "act": "work", "duration": 10},
            {"pid": 3, "act": "shop", "duration": 10},
        ]
    )
    assert novelty(hash_population(a), hash_population(b)) == 0.5
    assert conservatism(hash_population(a), hash_population(b)) == 0.5
