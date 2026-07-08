from typing import Iterable, Optional

import polars as pl
from robin.encoders.column_encoders.base import BaseEncoder
from robin.encoders.utils import inverse_token_frequency, tokenize
from torch import Tensor


class CategoricalTokeniser(BaseEncoder):
    def __init__(
        self, name: Optional[str] = None, verbose: bool = False, **kwargs
    ):
        """CategoricalEncoder is used to encode categorical data as integers from 0 to N.
        Where N is the number of unique categories.
        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
            verbose (bool, optional): print the encoder configuration. Defaults to False.
        Raises:
            UserWarning: If the data is not of type int or object.
        """
        self.name = name
        self.verbose = verbose
        self.encoding = "categorical"
        self.slot_size = 1

    def fit_and_encode(self, data: Iterable) -> tuple[Tensor, Tensor]:
        self._fit(data=data)
        return self.encode(data=data)

    def _fit(self, data: Iterable) -> Tensor:
        self.dtype = data.dtype
        encoded, self.mapping = tokenize(data)
        self._token_weights = inverse_token_frequency(encoded)
        self.size = len(self.mapping)

        if self.verbose:
            print(
                f">>> CategoricalTokeniser: Using token weights for {self.name} with shape {self._token_weights.shape} <<<"
            )
            if self.size > 20:
                print(
                    f">>> Warning: {self} has more than 20 categories ({self.size})). <<<"
                )

    def __str__(self):
        if self.verbose:
            return f"{self.__class__.__name__}: ({self.name}) size: {self.size}, categories: {self.mapping}, dtype: {self.dtype}"
        return f"{self.__class__.__name__}: ({self.name}) size: {self.size}"

    def encode(self, data: Iterable) -> tuple[Tensor, Tensor]:
        encoded = tokenize(data, self.mapping)[0]
        weights = self._token_weights[encoded.long()]
        return encoded.unsqueeze(-1), weights.unsqueeze(-1)

    def decode(self, data: Iterable, safe: bool = True) -> pl.Series:
        data = pl.Series(data.squeeze(-1)).cast(pl.Int8)
        reverse_mapping = {v: k for k, v in self.mapping.items()}
        if safe:
            missing = set(data.unique()) - set(reverse_mapping.keys())
            if missing:
                raise ValueError(
                    f"Missing categories in data: {missing}. Please check your encoding."
                )
        data = data.replace_strict(reverse_mapping, return_dtype=self.dtype)
        return data
