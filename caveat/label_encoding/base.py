from typing import Optional

import pandas as pd
import torch
from torch import Tensor

from caveat.label_encoding.label_weighting import (
    inverse_log_weight,
    inverse_weight,
    inverse_weights,
    log_inverse_weight,
    max_weight,
    product_weight,
    unit_weight,
    unit_weights,
)

label_weights_library = {"unit": unit_weights, "inverse": inverse_weights}

joint_label_weights_library = {
    "unit": unit_weight,
    "inverse": inverse_weight,
    "log_inverse": log_inverse_weight,
    "inverse_log": inverse_log_weight,
    "max_inverse": max_weight,
    "prod_inverse": product_weight,
}


class BaseLabelEncoder:
    def __init__(self, config: dict, **kwargs) -> None:
        """Base Attribute Encoder class."""
        self.config = config.copy()
        self.label_kwargs = {}
        self.weighting = kwargs.get("weighting", "unit")
        self.joint_weighting = kwargs.get("joint_weighting", "unit")

        self.label_weighter = label_weights_library.get(self.weighting, None)
        if self.label_weighter is None:
            raise ValueError(
                f"Unknown Label Encoder weighting: {self.label_weighting}, should be one of: {label_weights_library.keys()}"
            )

        self.joint_weighter = joint_label_weights_library.get(
            self.joint_weighting, None
        )
        if self.joint_weighter is None:
            raise ValueError(
                f"Unknown Label Encoder weighting: {self.joint_weighting}, should be one of: {joint_label_weights_library.keys()}"
            )

        print(
            f"""Label Encoder initialised with:
            Label weighting: {self.weighting}
            Joint weighting: {self.joint_weighting}
            """
        )

    def encode(self, data: pd.DataFrame) -> Tensor:
        raise NotImplementedError

    def decode(self, data: Tensor) -> pd.DataFrame:
        raise NotImplementedError


def ordinal_encode(data: pd.Series, min, max) -> Tensor:
    encoded = Tensor(data.values).unsqueeze(-1)
    encoded -= min
    encoded /= max - min
    return encoded.float()


def tokenize(data: pd.Series, encodings: Optional[dict] = None) -> Tensor:
    if encodings:
        missing = set(data.unique()) - set(encodings.keys())
        if missing:
            raise UserWarning(
                f"""
                Categories in data do not match existing categories: {missing}.
                Please specify the new categories in the encoding.
                Your existing encodings are: {encodings}
"""
            )
        nominals = pd.Categorical(data, categories=encodings.keys())
    else:
        nominals = pd.Categorical(data)
        encodings = {e: i for i, e in enumerate(nominals.categories)}
    nominals = torch.tensor(nominals.codes).int()
    return nominals, encodings


def onehot_encode(data: pd.Series, encodings: Optional[dict] = None) -> Tensor:
    nominals, encodings = tokenize(data, encodings)
    nominals = torch.nn.functional.one_hot(
        nominals.long(), num_classes=len(encodings)
    ).float()
    return nominals, encodings
