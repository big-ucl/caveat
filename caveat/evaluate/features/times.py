from numpy import array, ndarray
from pandas import DataFrame, Series

from caveat.evaluate.features.utils import weighted_features


def _act_enum_key(population: DataFrame) -> Series:
    """Compute act+cumcount key used by act_plan_enum features."""
    return population.act.astype(str) + population.groupby(
        ["pid", "act"], as_index=False, observed=False
    ).cumcount().astype(str)


def start_times_by_act(
    population: DataFrame, bin_size: int = 15, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    return weighted_features(
        population.groupby("act", observed=False).start.apply(list).to_dict(),
        bin_size=bin_size,
        factor=factor,
    )


def end_times_by_act(
    population: DataFrame, bin_size: int = 15, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    return weighted_features(
        population.groupby("act", observed=False).end.apply(list).to_dict(),
        bin_size=bin_size,
        factor=factor,
    )


def durations_by_act(
    population: DataFrame, bin_size: int = 15, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    return weighted_features(
        population.groupby("act", observed=False)
        .duration.apply(list)
        .to_dict(),
        bin_size=bin_size,
        factor=factor,
    )


def zip_columns(group, a: str = "start", b: str = "duration") -> ndarray:
    return array([(s, d) for s, d in zip(group[a], group[b])])


def start_durations_by_act(population: DataFrame) -> dict[str, ndarray]:
    if len(population) == 0:
        return {a: array([]) for a in population.act.unique()}
    sds = population.groupby("act", observed=False).apply(zip_columns).to_dict()
    return sds


def start_and_duration_by_act_bins(
    population: DataFrame, bin_size: int = 15, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    features = start_durations_by_act(population)
    return weighted_features(features, bin_size=bin_size, factor=factor)


def joint_durations_by_act_bins(
    population: DataFrame, bin_size: int = 15, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    if len(population) == 0:
        return {a: array([]) for a in population.act.unique()}
    transitions = population.reset_index()
    transitions = transitions.set_index(["index", "pid"])
    transitions.act = transitions.act.astype(str)
    transitions["shifted"] = transitions.duration.shift(-1)
    transitions = transitions.drop(transitions.groupby("pid").tail(1).index)
    transitions = (
        transitions.groupby("act", observed=False)
        .apply(zip_columns, a="duration", b="shifted")
        .to_dict()
    )
    return weighted_features(transitions, bin_size=bin_size, factor=factor)


def start_times_by_act_plan_seq(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = population.groupby("pid", as_index=False).cumcount().astype(
        str
    ) + population.act.astype(str)
    return weighted_features(
        population.groupby(actseq).start.apply(list).to_dict(), factor=1440
    )


def start_times_by_act_plan_enum(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = _act_enum_key(population)
    return weighted_features(
        population.groupby(actseq).start.apply(list).to_dict(), factor=1440
    )


def end_times_by_act_plan_seq(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = population.groupby("pid", as_index=False).cumcount().astype(
        str
    ) + population.act.astype(str)
    return weighted_features(
        population.groupby(actseq).end.apply(list).to_dict(), factor=1440
    )


def end_times_by_act_plan_enum(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = _act_enum_key(population)
    return weighted_features(
        population.groupby(actseq).end.apply(list).to_dict(), factor=1440
    )


def durations_by_act_plan_seq(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = population.groupby("pid", as_index=False).cumcount().astype(
        str
    ) + population.act.astype(str)
    return weighted_features(
        population.groupby(actseq).duration.apply(list).to_dict(), factor=1440
    )


def durations_by_act_plan_enum(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    actseq = _act_enum_key(population)
    return weighted_features(
        population.groupby(actseq).duration.apply(list).to_dict(), factor=1440
    )
