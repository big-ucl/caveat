from typing import List

import numpy as np
from numpy import array, ndarray
from pandas import DataFrame, MultiIndex, Series

from caveat.evaluate.features.utils import (
    _first_last_per_group,
    _grouped_sum,
    weighted_features,
)


def feasibility_eval(population: DataFrame, name: str) -> tuple[Series, Series]:
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

    if population.empty:
        print(f"Warning: {name} has no novel schedules for quality evaluation.")
        weights = Series([0] * len(index), index=index, name=f"{name}__weight")
        metrics = Series([0] * len(index), index=index, name=name)
        return weights, metrics

    pids = population.pid.values
    acts = population.act.values

    # home based feasibility
    unique_pids, first_idx, last_idx = _first_last_per_group(pids)
    first_acts = acts[first_idx]
    last_acts = acts[last_idx]

    not_start_at_home = first_acts != "home"
    not_end_at_home = last_acts != "home"
    not_home_based = not_start_at_home | not_end_at_home

    # consecutive feasibility
    consecutive_home = _get_consecutives(pids, acts, unique_pids, "home")
    consecutive_work = _get_consecutives(pids, acts, unique_pids, "work")
    consecutive_education = _get_consecutives(pids, acts, unique_pids, "education")

    consecutive = consecutive_home | consecutive_work | consecutive_education

    # combined
    invalid = not_home_based | consecutive

    n = len(unique_pids)

    metrics = Series(
        [
            invalid.sum() / n,
            not_home_based.sum() / n,
            not_start_at_home.sum() / n,
            not_end_at_home.sum() / n,
            consecutive.sum() / n,
            consecutive_home.sum() / n,
            consecutive_work.sum() / n,
            consecutive_education.sum() / n,
        ],
        index=index,
        name=name,
        dtype=float,
    )
    weights = Series(
        [n] * len(index), index=index, name=f"{name}__weight", dtype=int
    )
    return weights, metrics


def _get_consecutives(
    pids: ndarray, acts: ndarray, unique_pids: ndarray, target: str
) -> ndarray:
    """Check which pids have consecutive occurrences of target activity.

    Args:
        pids: 1D array of pid values (in schedule order).
        acts: 1D array of activity values (in schedule order).
        unique_pids: 1D array of unique pid values.
        target: Activity to check for consecutive occurrences.

    Returns:
        Boolean array of length len(unique_pids), True if pid has consecutive target.
    """
    is_target = acts == target
    # Previous row is same pid and also target
    prev_same_pid = np.empty(len(pids), dtype=bool)
    prev_same_pid[0] = False
    prev_same_pid[1:] = pids[1:] == pids[:-1]

    prev_is_target = np.empty(len(is_target), dtype=bool)
    prev_is_target[0] = False
    prev_is_target[1:] = is_target[:-1]

    consecutive = is_target & prev_is_target & prev_same_pid
    if consecutive.any():
        bad_pids = np.unique(pids[consecutive])
        return np.isin(unique_pids, bad_pids)
    return np.zeros(len(unique_pids), dtype=bool)


def get_consecutives(population: DataFrame, act: str) -> Series:
    """Legacy wrapper for backwards compatibility."""
    pids = population.pid.values
    acts = population.act.values
    unique_pids = np.unique(pids)
    result = _get_consecutives(pids, acts, unique_pids, act)
    return Series(result, index=unique_pids, name="consecutive")


def start_and_end_acts(
    population: DataFrame, target: str = "home"
) -> dict[str, tuple[ndarray, ndarray]]:
    pids = population.pid.values
    acts = population.act.values
    unique_pids, first_idx, last_idx = _first_last_per_group(pids)
    n = len(unique_pids)
    first = int((acts[first_idx] == target).sum())
    last = int((acts[last_idx] == target).sum())
    return {
        f"first act {target}": (array([0, 1]), array([(n - first), first])),
        f"last act {target}": (array([0, 1]), array([(n - last), last])),
    }


def act_consecutive(
    population: DataFrame, targets: List[str] = ["home", "work", "education"]
) -> dict[str, tuple[ndarray, ndarray]]:
    pids = population.pid.values
    acts = population.act.values
    unique_pids = np.unique(pids)
    n = len(unique_pids)
    result = {}
    for target in targets:
        consecutive_mask = _get_consecutives(pids, acts, unique_pids, target)
        c = int(consecutive_mask.sum())
        result[f"consecutive {target}"] = (
            array([0, 1]),
            array([n - c, c]),
        )
    return result


def contains_consecutive(schedule: DataFrame, act: str) -> int:
    mask = schedule.act.eq(act)
    consecutive = mask == mask.shift(1)
    return (mask & consecutive).sum() > 0


def time_consistency(
    population: DataFrame, target: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    pids = population.pid.values
    unique_pids, first_idx, last_idx = _first_last_per_group(pids)
    n = len(unique_pids)

    starts_val = population.start.values
    ends_val = population.end.values
    durations_val = population.duration.values

    starts = int((starts_val[first_idx] == 0).sum())
    ends = int((ends_val[last_idx] == target).sum())

    _, dur_sums = _grouped_sum(pids, durations_val)
    duration = int((dur_sums == target).sum())

    return {
        "starts at 0": (array([0, 1]), array([(n - starts), starts])),
        f"ends at {target}": (array([0, 1]), array([(n - ends), ends])),
        f"duration is {target}": (
            array([0, 1]),
            array([(n - duration), duration]),
        ),
    }


def duration_consistency(
    population: DataFrame, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    _, dur_sums = _grouped_sum(population.pid.values, population.duration.values)
    return weighted_features({"total duration": dur_sums / factor})


def sequence_lengths(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    _, pid_counts = np.unique(population.pid.values, return_counts=True)
    lengths, length_counts = np.unique(pid_counts, return_counts=True)
    return {"sequence lengths": (lengths, length_counts)}


def trip_consistency(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    raise NotImplementedError
