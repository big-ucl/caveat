from typing import Optional, Union

from numpy import array, ndarray, unique


def equals(
    a: dict[str, tuple[ndarray, ndarray]], b: dict[str, tuple[ndarray, ndarray]]
) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a.keys():
        if not len(a[k][0]) == len(b[k][0]):
            return False
        if not len(a[k][1]) == len(b[k][1]):
            return False
        if not (a[k][0] == b[k][0]).all():
            return False
        if not (a[k][1] == b[k][1]).all():
            return False
    return True


def bin_values(values: array, bin_size: Union[int, float]) -> ndarray:
    """
    Bins the input values based on the given bin size.

    Args:
        values (array): Input values to be binned.
        bin_size (int, float): Size of each bin.

    Returns:
        array: Binned values.
    """
    return (values // bin_size * bin_size) + (bin_size / 2)


def compress_feature(
    feature: list, bin_size: Optional[int] = None, factor: int = 1440
) -> tuple[ndarray, ndarray]:
    """
    Compresses a feature by optionally binning its values and returning unique values with counts.

    Args:
        feature (list): The feature to compress.
        bin_size (int, optional): The size of each bin. If None, no binning is performed.
        factor (int): Factor to apply to convert output values.

    Returns:
        tuple: A tuple containing two arrays and the total weight. The first array contains the unique
            values, and the second  array contains the counts of each value.
    """
    s = array(feature)
    if bin_size is not None:
        s = bin_values(s, bin_size)
    ks, ws = unique(s, axis=0, return_counts=True)
    ks = ks / factor
    return ks, ws


def weighted_features(
    features: dict[str, ndarray],
    bin_size: Optional[int] = None,
    factor: int = 1,
) -> dict[str, tuple[ndarray, ndarray]]:
    """
    Apply optional binning and value counting to dictionary of features.

    Args:
        features (dict[array): A dictionary of features to compress.
        bin_size (Optional[int]): The size of the bin to use for compression. Defaults to None.
        factor (int): Factor to apply to convert output values.

    Returns:
        dict[str, tuple[array, array[int]]]: A dictionary of features and weights.
    """
    return {
        k: compress_feature(values, bin_size, factor)
        for k, values in features.items()
    }
