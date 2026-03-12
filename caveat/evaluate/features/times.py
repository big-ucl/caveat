import numpy as np
from numpy import array, ndarray
from pandas import DataFrame

from caveat.evaluate.features.utils import _collect_by_group, _cumcount, weighted_features


def _act_enum_key(population: DataFrame) -> ndarray:
    """Compute act+cumcount key used by act_plan_enum features."""
    pids = population.pid.values
    acts = population.act.values
    _, pid_codes = np.unique(pids, return_inverse=True)
    _, act_codes = np.unique(acts, return_inverse=True)
    n_acts = len(np.unique(acts))
    compound = pid_codes * n_acts + act_codes
    cumcounts = _cumcount(compound)
    return np.array([str(a) + str(c) for a, c in zip(acts, cumcounts)], dtype=object)


def _seq_key(population: DataFrame) -> ndarray:
    """Compute cumcount+act key used by act_plan_seq features."""
    pids = population.pid.values
    acts = population.act.values
    cumcounts = _cumcount(pids)
    return np.array([str(c) + str(a) for c, a in zip(cumcounts, acts)], dtype=object)


def start_times_by_act(
    population: DataFrame, bin_size: int = 15, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    features = _collect_by_group(population.act.values, population.start.values)
    return weighted_features(features, bin_size=bin_size, factor=factor)


def end_times_by_act(
    population: DataFrame, bin_size: int = 15, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    features = _collect_by_group(population.act.values, population.end.values)
    return weighted_features(features, bin_size=bin_size, factor=factor)


def durations_by_act(
    population: DataFrame, bin_size: int = 15, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    features = _collect_by_group(population.act.values, population.duration.values)
    return weighted_features(features, bin_size=bin_size, factor=factor)


def start_durations_by_act(population: DataFrame) -> dict[str, ndarray]:
    if len(population) == 0:
        return {a: array([]) for a in population.act.unique()}
    acts = population.act.values
    pairs = np.column_stack([population.start.values, population.duration.values])
    return _collect_by_group(acts, pairs)


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
    pids = population.pid.values
    acts = population.act.values.astype(str)
    durations = population.duration.values.astype(float)

    # Identify rows that are not the last in their pid group
    same_pid_next = np.empty(len(pids), dtype=bool)
    same_pid_next[:-1] = pids[:-1] == pids[1:]
    same_pid_next[-1] = False

    # For valid rows, pair current duration with next duration
    valid_acts = acts[same_pid_next]
    valid_durs = durations[same_pid_next]
    next_durs = durations[1:][same_pid_next[:-1]]
    valid_pairs = np.column_stack([valid_durs, next_durs])

    features = _collect_by_group(valid_acts, valid_pairs)
    return weighted_features(features, bin_size=bin_size, factor=factor)


def start_times_by_act_plan_seq(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    keys = _seq_key(population)
    features = _collect_by_group(keys, population.start.values)
    return weighted_features(features, factor=1440)


def start_times_by_act_plan_enum(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    keys = _act_enum_key(population)
    features = _collect_by_group(keys, population.start.values)
    return weighted_features(features, factor=1440)


def end_times_by_act_plan_seq(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    keys = _seq_key(population)
    features = _collect_by_group(keys, population.end.values)
    return weighted_features(features, factor=1440)


def end_times_by_act_plan_enum(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    keys = _act_enum_key(population)
    features = _collect_by_group(keys, population.end.values)
    return weighted_features(features, factor=1440)


def durations_by_act_plan_seq(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    keys = _seq_key(population)
    features = _collect_by_group(keys, population.duration.values)
    return weighted_features(features, factor=1440)


def durations_by_act_plan_enum(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    keys = _act_enum_key(population)
    features = _collect_by_group(keys, population.duration.values)
    return weighted_features(features, factor=1440)
