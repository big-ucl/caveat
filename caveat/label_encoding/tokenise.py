from typing import List, Tuple

import pandas as pd
import torch
from torch import Tensor

from caveat.label_encoding.base import BaseLabelEncoder, row_probs, tokenize


class TokenAttributeEncoder(BaseLabelEncoder):

    def encode(self, data: pd.DataFrame) -> Tuple[Tensor, Tensor]:
        if not self.label_kwargs:
            # build config mappings and define label_kwargs
            self.build_config(data)
        return self._encode(data)

    def build_config(self, data: pd.DataFrame) -> Tensor:
        self.label_kwargs["label_embed_sizes"] = []

        for i, (k, v) in enumerate(self.config.copy().items()):
            if k not in data.columns:
                raise UserWarning(f"Conditional '{k}' not found in attributes")
            if isinstance(v, dict):  # Pre-defined encoding
                if v.get("ordinal"):
                    raise UserWarning(
                        "Ordinal encoding not supported for token encoding, change config to nominal or remove"
                    )
                elif v.get("nominal"):
                    if not isinstance(v.get("nominal"), dict):
                        raise UserWarning(
                            "Nominal encoding must be a dict of categories to index"
                        )
                    self.config[k].update(
                        {"location": i, "type": data[k].dtype}
                    )
                    self.label_kwargs["label_embed_sizes"].append(
                        data[k].nunique()
                    )
                else:
                    raise UserWarning(
                        f"Unrecognised attribute encoding in configuration: {v}"
                    )

            elif v == "nominal":  # Undefined nominal encoding
                _, nominal_encodings = tokenize(data[k], None)
                self.config[k] = {
                    "nominal": nominal_encodings,
                    "location": i,
                    "type": data[k].dtype,
                }
                self.label_kwargs["label_embed_sizes"].append(
                    len(nominal_encodings)
                )

            elif v == "ordinal":  # Undefined ordinal encoding
                raise UserWarning(
                    "Ordinal encoding not supported for token encoding, change config to nominal or remove"
                )

            else:
                raise UserWarning(
                    f"Unrecognised attribute encoding in configuration: {v}"
                )

    def _encode(self, data: pd.DataFrame) -> Tensor:
        encoded = []
        weights = []
        for k, v in self.config.items():
            if k not in data.columns:
                raise UserWarning(f"Conditional '{k}' not found in attributes")
            nominal_encoded, _ = tokenize(data[k], v["nominal"])
            encoded.append(nominal_encoded)
            freq = row_probs(data[k])
            inv_freq = 1 / freq
            weights.append(inv_freq)

        if not encoded:
            raise UserWarning("No attribute encoding found.")

        return (
            torch.stack(encoded, dim=-1).float(),
            torch.stack(weights, dim=-1).float(),
        )

    def decode(self, data: List[Tensor]) -> pd.DataFrame:
        decoded = {"pid": list(range(data.shape[0]))}
        for k, v in self.config.items():
            location, column_type = (v["location"], v["type"])
            if v.get("nominal") is not None:
                encoding = {i: name for name, i in v["nominal"].items()}
                tokens = data[:, location].tolist()
                decoded[k] = pd.Series([encoding[i] for i in tokens]).astype(
                    column_type
                )
            else:
                raise UserWarning(
                    f"Unrecognised attribute encoding in configuration: {k, v}"
                )

        return pd.DataFrame(decoded)

    def argmax_decode(self, data: List[Tensor]) -> pd.DataFrame:
        decoded = {"pid": list(range(data[0].shape[0]))}
        for k, v in self.config.items():
            location, column_type = (v["location"], v["type"])
            if v.get("nominal") is not None:
                encoding = {i: name for name, i in v["nominal"].items()}
                tokens = data[location].argmax(dim=-1).tolist()
                decoded[k] = pd.Series([encoding[i] for i in tokens]).astype(
                    column_type
                )
            else:
                raise UserWarning(
                    f"Unrecognised attribute encoding in configuration: {k, v}"
                )

        return pd.DataFrame(decoded)

    def sample_decode(self, data: List[Tensor]) -> pd.DataFrame:
        decoded = {"pid": list(range(data[0].shape[0]))}
        for k, v in self.config.items():
            location, column_type = (v["location"], v["type"])
            if v.get("nominal") is not None:
                encoding = {i: name for name, i in v["nominal"].items()}
                tokens = (
                    torch.multinomial(data[location], num_samples=1)
                    .flatten()
                    .tolist()
                )
                decoded[k] = pd.Series([encoding[i] for i in tokens]).astype(
                    column_type
                )
            else:
                raise UserWarning(
                    f"Unrecognised attribute encoding in configuration: {k, v}"
                )

        return pd.DataFrame(decoded)
