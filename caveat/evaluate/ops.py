import numpy as np
from numpy import ndarray
from pandas import Series


def actual(features: dict[str, float]) -> Series:
    return Series(features)


def feature_value(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series({k: v[0] for k, (v, w) in features.items()})


def feature_length(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series({k: len(v) for k, (v, w) in features.items()})


def feature_weight(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series({k: w.sum() for k, (v, w) in features.items()})


def average_weight(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series({k: w.mean() for k, (v, w) in features.items()})


def average_density(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    total = sum(w.sum() for _, w in features.values())
    return Series({k: w.sum() / total for k, (v, w) in features.items()})


def average(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    weighted_average = {}
    for k, (v, w) in features.items():
        if w.sum() > 0:
            weighted_average[k] = np.average(v, axis=0, weights=w).sum()
        else:
            weighted_average[k] = 0
    return Series(weighted_average)


def average2d(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series(
        {
            k: np.average(v, axis=0, weights=w).sum().sum()
            for k, (v, w) in features.items()
            if w.sum() > 0
        }
    )
