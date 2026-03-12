from typing import Optional, Union

import numpy as np
from numpy import array, ndarray, unique


def _collect_by_group(keys, values):
    """Group values by keys into a dict of arrays.

    Args:
        keys: 1D array of group keys.
        values: 1D or 2D array of values to group.

    Returns:
        dict mapping each unique key to the corresponding array of values.
    """
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_vals = values[order]
    unique_keys, start_idx = np.unique(sorted_keys, return_index=True)
    splits = np.split(sorted_vals, start_idx[1:])
    return {k: v for k, v in zip(unique_keys, splits)}


def _count_matrix(row_keys, col_keys):
    """Build a count matrix (crosstab) from row and column key arrays.

    Args:
        row_keys: 1D array of row group keys.
        col_keys: 1D array of column group keys.

    Returns:
        tuple of (matrix, unique_rows, unique_cols).
    """
    unique_rows, row_codes = np.unique(row_keys, return_inverse=True)
    unique_cols, col_codes = np.unique(col_keys, return_inverse=True)
    matrix = np.zeros((len(unique_rows), len(unique_cols)), dtype=np.int64)
    np.add.at(matrix, (row_codes, col_codes), 1)
    return matrix, unique_rows, unique_cols


def _cumcount(group_keys):
    """Compute cumulative count within each group (like groupby().cumcount()).

    Args:
        group_keys: 1D array of group keys.

    Returns:
        1D int array with the within-group ordinal for each element.
    """
    n = len(group_keys)
    order = np.argsort(group_keys, kind="stable")
    sorted_keys = group_keys[order]
    # Find group start positions in the sorted array
    change_positions = np.concatenate(
        [[0], np.where(sorted_keys[1:] != sorted_keys[:-1])[0] + 1]
    )
    counts = np.diff(np.concatenate([change_positions, [n]]))
    group_starts = np.repeat(change_positions, counts)
    sorted_cumcounts = np.arange(n) - group_starts
    # Map back to original order
    result = np.empty(n, dtype=np.int64)
    result[order] = sorted_cumcounts
    return result


def _grouped_sum(keys, values):
    """Sum values per group key.

    Args:
        keys: 1D array of group keys.
        values: 1D numeric array.

    Returns:
        tuple of (unique_keys, sums).
    """
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_vals = values[order]
    unique_keys, start_idx = np.unique(sorted_keys, return_index=True)
    sums = np.add.reduceat(sorted_vals, start_idx)
    return unique_keys, sums


def _first_last_per_group(keys):
    """Find the original indices of the first and last element per group.

    Args:
        keys: 1D array of group keys.

    Returns:
        tuple of (unique_keys, first_indices, last_indices) where indices
        refer to positions in the original array.
    """
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    unique_keys, start_idx = np.unique(sorted_keys, return_index=True)
    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:] - 1
    end_idx[-1] = len(keys) - 1
    return unique_keys, order[start_idx], order[end_idx]


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
