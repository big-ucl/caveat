from pandas import DataFrame

from caveat.evaluate.filters import filter_novel, no_filter


def test_filter_noop():
    scenario = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 5},
            {"pid": 1, "act": "work", "duration": 10},
            {"pid": 1, "act": "home", "duration": 15},
        ]
    )
    base = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 5},
            {"pid": 1, "act": "work", "duration": 10},
            {"pid": 1, "act": "home", "duration": 15},
        ]
    )
    assert no_filter(scenario, base).equals(scenario)


def test_filter_novel():
    scenario = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 5},
            {"pid": 1, "act": "visit", "duration": 10},
            {"pid": 1, "act": "home", "duration": 15},
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 2, "act": "work", "duration": 5},
            {"pid": 2, "act": "home", "duration": 15},
            {"pid": 3, "act": "home", "duration": 10},
            {"pid": 3, "act": "work", "duration": 5},
            {"pid": 3, "act": "home", "duration": 15},
        ]
    )
    base = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 5},
            {"pid": 1, "act": "other", "duration": 10},
            {"pid": 1, "act": "home", "duration": 15},
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 2, "act": "work", "duration": 10},
            {"pid": 2, "act": "home", "duration": 10},
        ]
    )
    filtered = filter_novel(scenario, base)
    print(filtered)
    assert filtered.equals(
        DataFrame(
            [
                {"pid": 1, "act": "home", "duration": 5},
                {"pid": 1, "act": "visit", "duration": 10},
                {"pid": 1, "act": "home", "duration": 15},
                {"pid": 2, "act": "home", "duration": 10},
                {"pid": 2, "act": "work", "duration": 5},
                {"pid": 2, "act": "home", "duration": 15},
                {"pid": 3, "act": "home", "duration": 10},
                {"pid": 3, "act": "work", "duration": 5},
                {"pid": 3, "act": "home", "duration": 15},
            ]
        )
    )
