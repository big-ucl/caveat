from typing import List

from numpy import array, ndarray
from pandas import DataFrame, MultiIndex, Series

from caveat.evaluate.features.utils import weighted_features


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

    # home based feasibility
    first_acts = population.groupby("pid").first().act
    last_acts = population.groupby("pid").last().act

    not_start_at_home = first_acts != "home"
    not_end_at_home = last_acts != "home"
    not_home_based = not_start_at_home | not_end_at_home

    # consecutive feasibility
    consecutive_home = get_consecutives(population, "home")
    consecutive_work = get_consecutives(population, "work")
    consecutive_education = get_consecutives(population, "education")

    consecutive = consecutive_home | consecutive_work | consecutive_education

    # combined
    invalid = not_home_based | consecutive

    n = population.pid.nunique()

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


def get_consecutives(population: DataFrame, act: str) -> Series:
    mask = population.act == act
    mask = mask & mask.shift(1)
    include = mask.groupby(population.pid).cumcount(ascending=True) > 0
    mask = mask & include
    mask = mask.groupby(population.pid).sum()
    return mask > 0


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
