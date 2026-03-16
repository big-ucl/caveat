import random
from typing import Optional

from pandas import DataFrame, Series


def sample_data(
    sequences: DataFrame, attributes: Optional[DataFrame], config: dict
) -> tuple:
    """Sample a proportion of the data based on sampler config.

    Args:
        sequences (DataFrame): input sequences data.
        attributes (Optional[DataFrame]): input attributes data.
        config (dict): configuration.

    Returns:
        Tuple[DataFrame, Optional[DataFrame]]: sampled sequences and attributes.
    """
    sequences = sample_sequences(sequences, config)
    if attributes is not None:
        attributes = select_attributes(attributes, sequences.pid)
    return sequences, attributes


def sample_sequences(data: DataFrame, config: dict) -> DataFrame:
    """Sample a proportion of the data based on sampler config.

    Args:
        data (DataFrame): input sequences data.
        config (dict): configuration.

    Returns:
        DataFrame: sampled sequences.
    """
    cnfg = config.get("sampler_params", {})
    sampler = cnfg.get("type")
    if sampler in [None, "none", "None", "NONE", ""]:
        return data
    elif sampler == "random":
        return random_sample(data, p=cnfg["p"])
    elif sampler == "biased":
        return biased_sample(data, p=cnfg["p"], threshold=cnfg["threshold"])
    else:
        raise ValueError(f"Sampler {sampler} not implemented.")


def select_attributes(data: DataFrame, idxs: Series) -> DataFrame:
    return data.copy().set_index("pid").loc[idxs.unique()].reset_index()


def random_sample(data: DataFrame, p: float) -> DataFrame:
    """Sample a proportion of the data.

    Args:
        data (DataFrame): input sequences data.
        p (float): proportion to sample.

    Returns:
        DataFrame: sampled sequences.
    """

    n_samples = int(len(data.pid.unique()) * p)
    sample_ids = random.sample(list(data.pid.unique()), n_samples)
    sampled = data[data.pid.isin(sample_ids)]
    return sampled


def biased_sample(data: DataFrame, p: float, threshold: int = 20) -> DataFrame:
    """
    Sample sequences that contain short activities according to the threshold.

    Args:
        data (DataFrame): input sequences data.
        p (float): proportion to sample.
        threshold (int): threshold to sample.

    Returns:
        DataFrame: sampled sequences.
    """
    candidates = data.groupby("pid").filter(
        lambda g: g.duration.min() < threshold
    )
    n_samples = int(candidates.pid.nunique() * (1 - p))  # to remove
    ids = random.sample(list(data.pid.unique()), n_samples)
    sampled = data[~data.pid.isin(ids)]
    return sampled
