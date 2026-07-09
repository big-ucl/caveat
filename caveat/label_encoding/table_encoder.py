from typing import List, Optional

import pandas as pd
import pandas.api.types as ptypes
from torch import Tensor, cat

from caveat.label_encoding.base import BaseLabelEncoder
from caveat.label_encoding.column_encoders.categorical import (
    CategoricalTokeniser,
)
from caveat.label_encoding.column_encoders.numeric import (
    MinMaxEncoder,
    # StandardScalerEncoder,
)


class TableEncoder(BaseLabelEncoder):
    continuous_encoder = MinMaxEncoder
    max_components: Optional[int] = (None,)
    learn_rounding_scheme: bool = (False,)
    enforce_min_max_values: bool = (False,)

    def fit_and_encode(self, data: pd.DataFrame) -> None:
        self.encoders = {}
        self.label_kwargs = {}
        self.label_kwargs["names"] = []
        self.label_kwargs["slot_sizes"] = []
        self.label_kwargs["label_embed_sizes"] = []
        self.mode = type(data)
        self.initialise_encoders(data)

        encoded = []
        for name, encoder in self.encoders.items():
            if name not in data.columns:
                raise ValueError(
                    f"Expected column '{name}' based on configuration, but not found in data"
                )
            x, _ = encoder.fit_and_encode(data[name])
            encoded.append(x)

        for name, encoder in self.encoders.items():
            self.label_kwargs["names"].append(name)
            self.label_kwargs["slot_sizes"].append(encoder.slot_size)
            self.label_kwargs["label_embed_sizes"].append(encoder.size)

        print(str(self))

        if not encoded:
            raise ValueError("No encodings found.")

        encoded = cat(encoded, dim=-1).float()
        weights = self.label_weighter(encoded)
        joint_weights = self.joint_weighter(encoded)
        return encoded, (weights, joint_weights)

    def encode(self, data: pd.DataFrame) -> Tensor:
        """Encode the dataframe into a Tensor.
        Args:
            data (pd.DataFrame): input dataframe to encode.
        Returns:
            Tensor: encoded dataframe.
        """
        encoded = []
        for column, encoder in self.encoders.items():
            if column not in data.columns:
                raise ValueError(
                    f"Expected column '{column}' based on configuration, but not found in data"
                )
            x, _ = encoder.encode(data[column])
            encoded.append(x)

        if not encoded:
            raise ValueError("No encodings found.")

        encoded = cat(encoded, dim=-1).float()
        weights = self.label_weighter(encoded)
        joint_weights = self.joint_weighter(encoded)
        return encoded, (weights, joint_weights)

    def decode(self, data: List[Tensor]) -> pd.DataFrame:
        """Decode Tensor of tokens back into dataframe.

        Args:
            data (List[Tensor]): input Tensor of tokens to decode.

        Returns:
            pd.DataFrame: decoded dataframe.
        """
        assert data.ndim == 2, "Data must be a 2D Tensor"
        assert data.shape[1] == sum(
            self.slot_sizes()
        ), "Data shape does not match encoder configuration"

        decoded = {}
        for (name, encoder), (i, j) in zip(
            self.encoders.items(), self.slot_idxs()
        ):
            tokens = data[:, i:j]
            decoded[name] = encoder.decode(tokens)
        decoded = pd.DataFrame(decoded)
        decoded.index.name = "pid"
        return decoded

    def __repr__(self):
        return f"{self.__class__.__name__}: ({len(self.encoders)} encoders)"

    def __str__(self):
        return f"{self.__repr__()}:\n" + "\n".join(
            [f"\t--> {e}" for e in self.encoders.values()]
        )

    def initialise_encoders(self, data: pd.DataFrame) -> None:
        if isinstance(data, pd.DataFrame):
            self.configure_pandas(data)
        else:
            raise ValueError("Data must be a pandas or polars dataframe")

    def configure_pandas(self, data: pd.DataFrame) -> None:
        """Configure the tokeniser by encoding the dataframe columns.
        Args:
            data (pd.DataFrame): input dataframe to configure.
        """
        for column in self.config.keys():
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in attributes")
            values = data[column]
            if (
                ptypes.is_string_dtype(values)
                or ptypes.is_object_dtype(values)
                or isinstance(values.dtype, pd.CategoricalDtype)
                or ptypes.is_bool_dtype(values)
                or len(set(values)) == 1
            ):
                self.encoders[column] = CategoricalTokeniser(name=column)
            elif ptypes.is_numeric_dtype(values):
                self.encoders[column] = self.continuous_encoder(
                    name=column, max_components=self.max_components
                )
            else:
                raise ValueError(
                    f"Column '{column}' not supported for encoding: {values.dtype}"
                )

    def encode_series(self, data: pd.Series) -> Tensor:
        """Encode a pandas series into a 1d Tensor.
        Args:
            data (pd.Series): input series to encode.
        Returns:
            Tensor: encoded series.
        """
        if data.name not in self.encoders.keys():
            raise ValueError(f"'{data.name}' not found in available encoders")
        encoder = self.encoders[data.name]
        column_encoded = encoder.encode(data)
        return column_encoded

    def names(self) -> List[str]:
        """Get the names of the encoders.
        Returns:
            List[str]: list of encoder names.
        """
        return list(self.encoders.keys())

    def types(self) -> List[str]:
        """Get the types of the embeddings.
        Returns:
            List[str]: list of types of the embeddings.
        """
        return [encoder.encoding for encoder in self.encoders.values()]

    def slot_sizes(self) -> List[int]:
        """Get the slot sizes of the embeddings.
        Returns:
            List[int]: list of slot sizes of the embeddings.
        """
        return [encoder.slot_size for encoder in self.encoders.values()]

    def slot_idxs(self) -> List[int]:
        """Get the slot locations of the embeddings.
        Returns:
            List[int]: list of slot locations of the embeddings.
        """
        idxs = [0]
        for s in self.slot_sizes():
            idxs.append(idxs[-1] + s)
        starts = idxs[:-1]
        ends = idxs[1:]
        return list(zip(starts, ends))

    def sizes(self) -> List[int]:
        """Get the sizes of the embeddings.
        Returns:
            List[int]: list of sizes of the embeddings.
        """
        return [encoder.size for encoder in self.encoders.values()]

    def token_weights(self) -> list[Tensor]:
        """Get the token weights for each encoder."""
        return [encoder._token_weights for encoder in self.encoders.values()]
