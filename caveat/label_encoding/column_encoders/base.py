import abc
import sys
from typing import Iterable, Optional

import polars as pl
from torch import Tensor

MAX_DECIMALS = sys.float_info.dig


class BaseEncoder(abc.ABC):
    dtype = None
    encoding = None
    size = None
    _rounding_digits = None
    _token_weights = None

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def read_polars(
        self, data: Iterable, name: Optional[str] = None
    ) -> pl.Series:
        if not isinstance(data, pl.Series):
            print(
                f"Attempting to convert data ({type(data)}) to polars Series."
            )
            data = pl.Series(data)

        return data

    @abc.abstractmethod
    def fit_and_encode(self, data: Iterable) -> Tensor:
        """Fit the encoder to the data.

        Args:
            data (Iterable): input data to fit to.

        Returns:
            Tensor: token weights.
        """

    @abc.abstractmethod
    def encode(self, data: Iterable) -> Tensor:
        """Encode the data. Returns encoded data and weights.

        Args:
            data (Iterable): input data to be encoded.

        Returns:
            tuple[Tensor, Tensor]: encoded data, token_weights.
        """

    @abc.abstractmethod
    def decode(self, data: Iterable) -> pl.Series:
        """Decode the data.

        Args:
            data (Iterable): input data to be decoded.

        Returns:
            pl.Series: decoded data.
        """

    def get_rounding(self, data: pl.Series) -> Optional[int]:
        """Learn the number of digits to round data to.

        Args:
            data (pl.Series):
                Data to learn the number of digits to round to.

        Returns:
            int or None:
                Number of digits to round to.
        """

        roundable_data = data.filter(data.is_finite())
        if len(roundable_data) == 0:
            name = data.name if data.name is not None else "unknown"
            print(
                f"No finite data found for column '{name}'. Cannot learn rounding scheme."
            )
            return None

        # Try to round to fewer digits
        highest_int = int(roundable_data.abs().max())
        most_digits = len(str(highest_int)) if highest_int != 0 else 0
        max_decimals = max(0, MAX_DECIMALS - most_digits)
        if (roundable_data == roundable_data.round(max_decimals)).all():
            for decimal in range(max_decimals + 1):
                if (roundable_data == roundable_data.round(decimal)).all():
                    return decimal

        # Can't round, not equal after MAX_DECIMALS digits of precision
        name = data.name if data.name is not None else "unknown"
        print(
            f"No rounding scheme detected for column '{name}'. Data will not be rounded."
        )
        return None
