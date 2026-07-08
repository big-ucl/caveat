from typing import Iterable, Optional

import polars as pl
from robin.encoders.column_encoders.base import BaseEncoder
from torch import Tensor, ones_like


class MinMaxEncoder(BaseEncoder):
    def __init__(
        self,
        name: Optional[str] = None,
        verbose: bool = False,
        learn_rounding: bool = False,
        **kwargs,
    ):
        """ContinuousEncoder is used to encode continuous data to a range between -1 and 1.

        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
            verbose (bool, optional): print the encoder configuration. Defaults to False.
        Raises:
            UserWarning: If the data is not of type int or float.
        """

        self.name = name
        self.verbose = verbose
        self._learn_rounding = learn_rounding
        self.encoding = "continuous"
        self.slot_size = 1
        self.size = 1

    def fit_and_encode(self, data: Iterable) -> Tensor:
        if not data.dtype.is_numeric():
            raise ValueError(
                "ContinuousEncoder only supports numeric data types."
            )

        self.mini = data.min()
        self.maxi = data.max()
        self.range = self.maxi - self.mini
        self.mean = data.mean()
        self.std = data.std()
        if self.range == 0:
            raise ValueError("Data has no range. Cannot encode.")
        self.dtype = data.dtype

        if self._learn_rounding:
            self._rounding_digits = self.get_rounding(data)

        return self.encode(data)

    def __str__(self):
        return f"{self.__class__.__name__}: ({self.name}) min: {self.mini}, max: {self.maxi}, range: {self.range}, dtype: {self.dtype}"

    def encode(self, data: Iterable) -> Tensor:
        data = Tensor(data).unsqueeze(-1)
        encoded = (2 * (data - self.mini) / self.range) - 1
        weights = ones_like(encoded)
        return encoded, weights

    def decode(self, data: Iterable) -> pl.Series:
        data = pl.Series(data.squeeze(-1))
        data = ((data + 1.0) * self.range / 2.0) + self.mini
        data = data.cast(self.dtype)
        if self._learn_rounding and self._rounding_digits is not None:
            data = data.round(self._rounding_digits)
        return data


class StandardScalerEncoder(BaseEncoder):
    def __init__(
        self,
        name: Optional[str] = None,
        verbose: bool = False,
        learn_rounding: bool = False,
        **kwargs,
    ):
        self.name = name
        self.verbose = verbose
        self._learn_rounding = learn_rounding
        self.encoding = "continuous"
        self.slot_size = 1
        self.size = 1

    def fit_and_encode(self, data: Iterable) -> Tensor:
        if not data.dtype.is_numeric():
            raise ValueError(
                "StandardScalerEncoder only supports numeric data types."
            )

        self.mean = data.mean()
        self.std = data.std()
        if self.std == 0:
            raise ValueError("Data has no variance. Cannot encode.")
        self.dtype = data.dtype

        if self._learn_rounding:
            self._rounding_digits = self.get_rounding(data)

        return self.encode(data)

    def __str__(self):
        return f"{self.__class__.__name__}: ({self.name}) mean: {self.mean}, std: {self.std}, dtype: {self.dtype}"

    def encode(self, data: Iterable) -> Tensor:
        data = Tensor(data).unsqueeze(-1)
        encoded = 0.25 * (data - self.mean) / self.std
        weights = ones_like(encoded)
        return encoded, weights

    def decode(self, data: Iterable) -> pl.Series:
        data = pl.Series(data.squeeze(-1))
        data = (data * self.std) * 4 + self.mean
        data = data.cast(self.dtype)
        if self._learn_rounding and self._rounding_digits is not None:
            data = data.round(self._rounding_digits)
        return data
