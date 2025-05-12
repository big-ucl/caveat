from typing import List

from numpy import array, ndarray
from pandas import DataFrame, MultiIndex, Series

from caveat.evaluate.features.utils import weighted_features


def feasibility_eval(
    population: DataFrame, name: str
) -> dict[str, tuple[ndarray, ndarray]]:
    index = MultiIndex.from_tuples(
        [
            ("sample quality", "invalid", "all"),
            ("sample quality", "not home based", "all"),
            ("sample quality", "not home based", "starts"),
            ("sample quality", "not home based", "ends"),
            ("sample quality", "consecutive", "all"),
            ("sample quality", "consecutive", "home"),
            # ("sample quality", "consecutive", "work"),
            # ("sample quality", "consecutive", "education"),
        ],
        names=["domain", "feature", "segment"],
    )

    if population.empty:
        print(f"Warning: {name} has no novel schedules for quality evaluation.")
        weights = Series([0] * len(index), index=index, name=f"{name}__weight")
        metrics = Series([0] * len(index), index=index, name=name)
        return weights, metrics

    n = population.pid.nunique()

    invalid = 0
    not_home_based = 0
    consecutives = 0
    not_start_at_home = 0
    not_end_at_home = 0
    consecutive_home = 0
    # consecutive_work = 0
    # consecutive_education = 0

    for _, schedule in population.groupby("pid"):
        nsh = schedule.act.iloc[0] != "home"
        neh = schedule.act.iloc[-1] != "home"
        nhb = any([nsh, neh])

        ch = contains_consecutive(schedule, "home")
        # cw = contains_consecutive(schedule, "work")
        # ce = contains_consecutive(schedule, "education")
        # ccs = any([ch, cw, ce])
        ccs = ch

        invalid += any([nhb, ccs])
        not_home_based += nhb
        not_start_at_home += nsh
        not_end_at_home += neh
        consecutives += ccs
        consecutive_home += ch
        # consecutive_work += cw
        # consecutive_education += ce

    metrics = Series(
        [
            invalid / n,
            not_home_based / n,
            not_start_at_home / n,
            not_end_at_home / n,
            consecutives / n,
            consecutive_home / n,
            # consecutive_work / n,
            # consecutive_education / n,
        ],
        index=index,
        name=name,
    )
    weights = Series([n] * len(index), index=index, name=f"{name}__weight")

    return weights, metrics


def start_and_end_acts(
    population: DataFrame, target: str = "home"
) -> dict[str, tuple[ndarray, ndarray]]:
    n = population.pid.nunique()
    first = (population.groupby("pid").first().act == target).sum()
    last = (population.groupby("pid").last().act == target).sum()
    return {
        f"first act {target}": (array([0, 1]), array([(n - first), first])),
        f"last act {target}": (array([0, 1]), array([(n - last), last])),
    }


def act_consecutive(
    population: DataFrame, targets: List[str] = ["home", "work", "education"]
) -> dict[str, tuple[ndarray, ndarray]]:
    result = {}
    n = population.pid.nunique()
    for target in targets:
        consecutive = (
            population.groupby("pid")
            .apply(lambda x: contains_consecutive(x, target))
            .sum()
        )
        result[f"consecutive {target}"] = (
            array([0, 1]),
            array([n - consecutive, consecutive]),
        )
    return result


def contains_consecutive(schedule: DataFrame, act: str) -> int:
    mask = schedule.act.eq(act)
    consecutive = mask == mask.shift(1)
    return (mask & consecutive).sum() > 0


def time_consistency(
    population: DataFrame, target: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    n = population.pid.nunique()
    starts = (population.groupby("pid").first().start == 0).sum()
    ends = (population.groupby("pid").last().end == target).sum()
    duration = (population.groupby("pid").duration.sum() == target).sum()
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
    durations = population.groupby("pid").duration.sum() / factor
    return weighted_features({"total duration": durations.array})


def sequence_lengths(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    lengths = population.groupby("pid").size().value_counts().sort_index()
    keys = array(lengths.index)
    values = array(lengths.values)
    return {"sequence lengths": (keys, values)}


def trip_consistency(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    raise NotImplementedError
