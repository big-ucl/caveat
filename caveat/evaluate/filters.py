from pandas import DataFrame

from caveat.evaluate.features import creativity


def no_filter(scenario: DataFrame, base: DataFrame) -> DataFrame:
    return scenario


def filter_novel(scenario: DataFrame, base: DataFrame) -> DataFrame:
    base_hashed = creativity.hash_population(base)
    # Vectorized: hash all synthetic schedules at once
    act_hash = scenario.act.astype(str) + scenario.duration.astype(str)
    synth_hashes = act_hash.groupby(scenario.pid).agg("".join)
    novel_pids = synth_hashes.index[~synth_hashes.isin(base_hashed)]
    return scenario[scenario.pid.isin(novel_pids)].reset_index(drop=True)
