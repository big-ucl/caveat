import pandas as pd
import torch
from torch import Tensor

from caveat.label_encoding.base import (
    BaseLabelEncoder,
    onehot_encode,
    ordinal_encode,
)


class OneHotAttributeEncoder(BaseLabelEncoder):

    def encode(self, data: pd.DataFrame) -> Tensor:
        i = 0
        encoded = []
        for k, v in self.config.copy().items():
            if k not in data.columns:
                raise UserWarning(f"Conditional '{k}' not found in attributes")

            if isinstance(v, dict):  # Pre-defined encoding
                if v.get("ordinal"):
                    if not isinstance(v.get("ordinal"), list):
                        raise UserWarning(
                            "Ordinal encoding must be a list of (min, max)"
                        )
                    self.validate_previous(k, v, i, 1, data[k])
                    min, max = v["ordinal"]
                    encoded.append(ordinal_encode(data[k], min, max))
                    self.config[k].update(
                        {"location": i, "length": 1, "type": data[k].dtype}
                    )
                    i += 1
                elif v.get("nominal"):
                    if not isinstance(v.get("nominal"), dict):
                        raise UserWarning(
                            "Nominal encoding must be a dict of categories to index"
                        )
                    self.validate_previous(k, v, i, len(v["nominal"]), data[k])
                    nominal_encoded, _ = onehot_encode(data[k], v["nominal"])
                    encoded.append(nominal_encoded)
                    self.config[k].update(
                        {
                            "location": i,
                            "length": len(v["nominal"]),
                            "type": data[k].dtype,
                        }
                    )
                    i += len(v["nominal"])
                else:
                    raise UserWarning(
                        f"Unrecognised attribute encoding in configuration: {v}"
                    )

            elif v == "nominal":  # Undefined nominal encoding
                nominal_encoded, nominal_encodings = onehot_encode(
                    data[k], None
                )
                encoded.append(nominal_encoded)
                self.config[k] = {
                    "nominal": nominal_encodings,
                    "location": i,
                    "length": len(nominal_encodings),
                    "type": data[k].dtype,
                }
                i += len(nominal_encodings)

            elif v == "ordinal":  # Undefined ordinal encoding
                min = data[k].min()
                max = data[k].max()
                encoded.append(ordinal_encode(data[k], min, max))
                self.config[k] = {
                    "ordinal": [min, max],
                    "location": i,
                    "length": 1,
                    "type": data[k].dtype,
                }
                i += 1

            else:
                raise UserWarning(
                    f"Unrecognised attribute encoding in configuration: {v}"
                )

        if not encoded:
            raise UserWarning("No attribute encodeding found.")

        encoded = torch.cat(encoded, dim=-1)

        # weighting
        weights = self.label_weighter(encoded)
        joint_weights = self.joint_weighter(encoded)

        return encoded, (weights, joint_weights)

    def validate_previous(self, k, v, i, expected_length, data) -> None:
        prev_location, prev_length, prev_type = (
            v.get("location"),
            v.get("length"),
            v.get("type"),
        )
        if prev_location is not None and prev_location != i:
            raise UserWarning(
                f"Ordinal encoding location mismatch for {k}: {prev_location} != {i}"
            )
        if prev_length is not None and prev_length != expected_length:
            raise UserWarning(
                f"Ordinal encoding length mismatch for {k}: {prev_length} != {expected_length}"
            )
        if prev_type is not None and prev_type != data.dtype:
            raise UserWarning(
                f"Ordinal encoding type mismatch for {k}: {prev_type} != {data.dtype}"
            )

    def decode(self, data: Tensor) -> pd.DataFrame:
        decoded = {"pid": list(range(data.shape[0]))}
        for k, v in self.config.items():
            location, length, column_type = (
                v["location"],
                v["length"],
                v["type"],
            )
            if v.get("ordinal") is not None:
                min, max = v["ordinal"]
                decoded[k] = pd.Series(
                    data[:, location] * (max - min) + min
                ).astype(column_type)
            elif v.get("nominal") is not None:
                encoding = {i: name for name, i in v["nominal"].items()}
                decoded[k] = pd.Series(
                    [
                        encoding[i]
                        for i in data[:, location : location + length]
                        .argmax(dim=-1)
                        .tolist()
                    ]
                ).astype(column_type)
            else:
                raise UserWarning(
                    f"Unrecognised attribute encoding in configuration: {k, v}"
                )

        return pd.DataFrame(decoded)
