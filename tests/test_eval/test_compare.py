from pandas import DataFrame

from caveat.evaluate.evaluate import Evaluator, compare, evaluate


def _observed():
    return DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "work", "start": 6, "end": 14, "duration": 8},
            {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
            {"pid": 1, "act": "home", "start": 0, "end": 10, "duration": 10},
            {"pid": 1, "act": "work", "start": 10, "end": 24, "duration": 14},
        ]
    )


def _synthetic():
    return DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "shop", "start": 6, "end": 14, "duration": 8},
            {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
            {"pid": 1, "act": "home", "start": 0, "end": 12, "duration": 12},
            {"pid": 1, "act": "work", "start": 12, "end": 24, "duration": 12},
        ]
    )


def test_compare_single_df():
    result = compare(_observed(), _synthetic(), report_stats=False)
    expected_keys = {
        "descriptions",
        "group_descriptions",
        "domain_descriptions",
        "distances",
        "group_distances",
        "domain_distances",
    }
    assert set(result.keys()) == expected_keys


def test_compare_dict():
    result = compare(
        _observed(), {"m1": _synthetic(), "m2": _synthetic()}, report_stats=False
    )
    # both model names should appear in output columns
    for frame in result.values():
        cols = set(frame.columns)
        assert "m1" in cols or any("m1" in c for c in cols)
        assert "m2" in cols or any("m2" in c for c in cols)


def test_compare_equals_evaluate():
    obs = _observed()
    syn = _synthetic()
    compare_result = compare(obs, {"m": syn}, report_stats=False)
    evaluate_result = evaluate({"m": syn}, obs, report_stats=False)
    for key in evaluate_result:
        assert compare_result[key].equals(evaluate_result[key]), (
            f"Mismatch in {key}"
        )


def test_evaluator_caches():
    obs = _observed()
    evaluator = Evaluator(obs)
    result1 = evaluator.compare({"m": _synthetic()}, report_stats=False)
    result2 = evaluator.compare({"m": _synthetic()}, report_stats=False)
    for key in result1:
        assert result1[key].equals(result2[key]), f"Mismatch in {key}"
