from numpy import ndarray
from pandas import DataFrame, MultiIndex, Series

from caveat.evaluate.features.utils import weighted_features


def transitions_by_act(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    transitions = population.reset_index()
    transitions = transitions.set_index(["index", "pid"])
    transitions.act = transitions.act.astype(str)
    transitions = transitions.act + ">" + transitions.act.shift(-1)
    transitions = transitions.drop(transitions.groupby("pid").tail(1).index)
    transitions = (
        transitions.groupby("pid")
        .value_counts()
        .unstack()
        .fillna(0)
        .astype(int)
        .to_dict(orient="list")
    )
    return weighted_features(transitions)


def transition_3s_by_act(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    transitions = population.reset_index()
    transitions = transitions.set_index(["index", "pid"])
    transitions.act = transitions.act.astype(str)
    transitions = (
        transitions.act
        + ">"
        + transitions.act.shift(-1)
        + ">"
        + transitions.act.shift(-2)
    )
    transitions = transitions.drop(transitions.groupby("pid").tail(2).index)
    transitions = (
        transitions.groupby("pid")
        .value_counts()
        .unstack()
        .fillna(0)
        .astype(int)
        .to_dict(orient="list")
    )
    return weighted_features(transitions)


def transition_4s_by_act(
    population: DataFrame,
) -> dict[str, tuple[ndarray, ndarray]]:
    transitions = population.reset_index()
    transitions = transitions.set_index(["index", "pid"])
    transitions.act = transitions.act.astype(str)
    transitions = (
        transitions.act
        + ">"
        + transitions.act.shift(-1)
        + ">"
        + transitions.act.shift(-2)
        + ">"
        + transitions.act.shift(-3)
    )
    transitions = transitions.drop(transitions.groupby("pid").tail(3).index)
    transitions = (
        transitions.groupby("pid")
        .value_counts()
        .unstack()
        .fillna(0)
        .astype(int)
        .to_dict(orient="list")
    )
    return weighted_features(transitions)


def tour(acts: Series) -> str:
    """
    Extracts the tour from the given Series of activities.

    Args:
        acts (Series): A Series containing the activities.

    Returns:
        str: A string representation of the tour.
    """
    return ">".join(acts.str[0])


def full_sequences(population: DataFrame) -> dict[str, tuple[ndarray, ndarray]]:
    transitions = population.reset_index()
    transitions = transitions.set_index(["index", "pid"])
    transitions.act = transitions.act.astype(str)
    transitions = transitions.groupby("pid").act.apply(tour)
    transitions = (
        transitions.groupby("pid")
        .value_counts()
        .unstack()
        .fillna(0)
        .astype(int)
        .to_dict(orient="list")
    )
    return weighted_features(transitions)


def collect_sequence(acts: Series) -> str:
    return ">".join(acts)


def sequence_probs(population: DataFrame) -> DataFrame:
    """
    Calculates the sequence probabilities in the given population DataFrame.

    Args:
        population (DataFrame): A DataFrame containing the population data.

    Returns:
        DataFrame: A DataFrame containing the probability of each sequence.
    """
    metrics = (
        population.groupby("pid")
        .act.apply(collect_sequence)
        .value_counts(normalize=True)
    )
    metrics = metrics.sort_values(ascending=False)
    metrics.index = MultiIndex.from_tuples(
        [("sequence rate", acts) for acts in metrics.index]
    )
    return metrics
