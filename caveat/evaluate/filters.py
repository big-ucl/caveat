from pandas import DataFrame

from caveat.evaluate.features import creativity


def no_filter(scenario: DataFrame, base: DataFrame) -> DataFrame:
    return scenario


def filter_novel(scenario: DataFrame, base: DataFrame) -> DataFrame:
    base_hashed = creativity.hash_population(base)
    filtered = scenario.groupby("pid", group_keys=False).apply(
        lambda x: x if creativity.hash_schedule(x) not in base_hashed else None
    )
    return filtered.reset_index(drop=True)
