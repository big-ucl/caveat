from pandas import DataFrame


def hash_schedule(schedule: DataFrame) -> list[str]:
    """Hash a schedule. We first create strings of combined activities and durations.

    Args:
        schedule (DataFrame): Input schedule.

    Returns:
        str: hashed schedule.
    """
    act_hash = schedule.act.astype(str) + schedule.duration.astype(str)
    return "".join(act_hash)


def hash_population(population: DataFrame) -> set[str]:
    """Hash a population of sequences. We first create strings of combined activities and durations.
    Then create a python set of these strings. This will remove duplicates.

    Args:
        population (DataFrame): Input population of sequences.

    Returns:
        set[str]: set of hashed sequences.
    """
    act_hash = population.act.astype(str) + population.duration.astype(str)
    return set(act_hash.groupby(population.pid).apply("".join))


def diversity(population: DataFrame, hashed: set[str]) -> float:
    """Measure the internal diversity of a population of sequences. This is the ratio of unique
    sequences to the total number of sequences.

    Args:
        population (DataFrame): Input population of sequences.
        hashed (set[str]): Hashed population of sequences.

    Returns:
        float: Diversity of the population.
    """
    n = population.pid.nunique()
    if n == 0:
        return 0
    unique = len(hashed)
    return unique / n


def homogeneity(population: DataFrame, hashed: set[str]) -> float:
    """Measure the internal homogeneity of a population of sequences. This is 1-diversity.

    Args:
        population (DataFrame): Input population of sequences.
        hashed (set[str]): Hashed population of sequences.

    Returns:
        float: Homogeneity of the synthetic sample.
    """
    return 1 - diversity(population, hashed)


def novelty(observed_hashed: set[str], synthetic_hashed: set[str]) -> float:
    """Measure the novelty of a population by comparing it to an observed population.

    Args:
        observed_hashed (set[str]): Hashed observed population.
        synthetic_hashed (set[str]): Hashed synthetic population.

    Returns:
        float: Novelty of the synthetic population.
    """
    n = len(synthetic_hashed)
    if n == 0:
        return 0
    return len(synthetic_hashed - observed_hashed) / n


def conservatism(
    observed_hashed: set[str], synthetic_hashed: set[str]
) -> float:
    """Measure the conservatism of a population as 1-novelty.

    Args:
        observed_hashed (set[str]): Hashed observed population.
        synthetic_hashed (set[str]): Hashed synthetic population.

    Returns:
        float: Conservatism of the synthetic population.
    """
    return 1 - novelty(observed_hashed, synthetic_hashed)
