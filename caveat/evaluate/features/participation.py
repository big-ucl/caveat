import numpy as np
from numpy import array, ndarray
from pandas import DataFrame

from caveat.evaluate.features.utils import (
    _collect_by_group,
    _count_matrix,
    _cumcount,
    weighted_features,
)


def participation_prob_by_act(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    """Calculate the participations by activity for a given population.

    Args:
        population (DataFrame): The population data.

    Returns:
        dict[str, tuple[array, array]]: A dictionary containing the participation for each activity.
    """
    pids = population.pid.values
    acts = population.act.values
    matrix, unique_pids, unique_acts = _count_matrix(pids, acts)
    participated = (matrix > 0).sum(axis=0)
    n = len(unique_pids)
    return {
        act: (array([0, 1]), array([n - participated[j], participated[j]]))
        for j, act in enumerate(unique_acts)
    }


def participation_rates(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    _, counts = np.unique(population.pid.values, return_counts=True)
    return weighted_features({"all": counts})


def participation_rates_by_act(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    pids = population.pid.values
    acts = population.act.values
    matrix, _, unique_acts = _count_matrix(pids, acts)
    return weighted_features(
        {act: matrix[:, j] for j, act in enumerate(unique_acts)}
    )


def participation_rates_by_seq_act(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    pids = population.pid.values
    acts = population.act.values
    cumcounts = _cumcount(pids)
    # Build composite key: "0home", "1work", etc.
    keys = np.array(
        [str(c) + str(a) for c, a in zip(cumcounts, acts)], dtype=object
    )
    matrix, _, unique_keys = _count_matrix(pids, keys)
    return weighted_features(
        {k: matrix[:, j] for j, k in enumerate(unique_keys)}
    )


def participation_rates_by_act_enum(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    pids = population.pid.values
    acts = population.act.values
    # Cumcount within (pid, act) groups using compound integer key
    _, pid_codes = np.unique(pids, return_inverse=True)
    _, act_codes = np.unique(acts, return_inverse=True)
    n_acts = len(np.unique(acts))
    compound = pid_codes * n_acts + act_codes
    cumcounts = _cumcount(compound)
    # Build composite key: "home0", "work1", etc.
    keys = np.array(
        [str(a) + str(c) for a, c in zip(acts, cumcounts)], dtype=object
    )
    matrix, _, unique_keys = _count_matrix(pids, keys)
    return weighted_features(
        {k: matrix[:, j] for j, k in enumerate(unique_keys)}
    )


def calc_pair_prob(act_counts, pair):
    a, b = pair
    if a == b:
        return (act_counts[a] > 1).sum()
    return ((act_counts[a] > 0) & (act_counts[b] > 0)).sum()


def calc_pair_rate(act_counts, pair):
    a, b = pair
    if a == b:
        return ((act_counts[a] / 2).astype(int)).value_counts().to_dict()
    return (
        ((act_counts[[a, b]].min(axis=1) / 2).astype(int))
        .value_counts()
        .to_dict()
    )


def combinations_with_replacement(
    targets: list, length: int, prev_array=[]
) -> list[list]:
    """Returns all possible combinations of elements in the input array with replacement,
    where each combination has a length of tuple_length.

    Args:
        targets (list): The input array to generate combinations from.
        length (int): The length of each combination.
        prev_array (list, optional): The previous array generated in the recursion. Defaults to [].

    Returns:
        list: A list of all possible combinations of elements in the input array with replacement.
    """
    if len(prev_array) == length:
        return [prev_array]
    combs = []
    for i, val in enumerate(targets):
        prev_array_extended = prev_array.copy()
        prev_array_extended.append(val)
        combs += combinations_with_replacement(
            targets[i:], length, prev_array_extended
        )
    return combs


def joint_participation_prob(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    """Calculate the participation prob for all pairs of activities in the given population.

    Args:
        population (pandas.DataFrame): A DataFrame containing the population data.

    Returns:
        dict: A dictionary containing the participation probability for all pairs of activities.
    """
    pids = population.pid.values
    acts = population.act.values
    matrix, _, unique_acts = _count_matrix(pids, acts)
    act_list = list(unique_acts)
    act_idx = {a: i for i, a in enumerate(act_list)}
    n = matrix.shape[0]
    pairs = combinations_with_replacement(act_list, 2)
    metric = {}
    for pair in pairs:
        ai, bi = act_idx[pair[0]], act_idx[pair[1]]
        if pair[0] == pair[1]:
            p = int((matrix[:, ai] > 1).sum())
        else:
            p = int(((matrix[:, ai] > 0) & (matrix[:, bi] > 0)).sum())
        metric["+".join(pair)] = (array([0, 1]), array([n - p, p]))
    return metric


def joint_participation_rate(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    """Calculate the participation rate for all pairs of activities in the given population.

    Args:
        population (pandas.DataFrame): A DataFrame containing the population data.

    Returns:
        dict: A dictionary containing the participation rate for all pairs of activities.
    """
    pids = population.pid.values
    acts = population.act.values
    matrix, _, unique_acts = _count_matrix(pids, acts)
    act_list = list(unique_acts)
    act_idx = {a: i for i, a in enumerate(act_list)}
    pairs = combinations_with_replacement(act_list, 2)
    metric = {}
    for pair in pairs:
        ai, bi = act_idx[pair[0]], act_idx[pair[1]]
        if pair[0] == pair[1]:
            vals = matrix[:, ai] // 2
        else:
            vals = np.minimum(matrix[:, ai], matrix[:, bi]) // 2
        keys, counts = np.unique(vals, return_counts=True)
        metric["+".join(pair)] = (keys, counts)
    return metric
