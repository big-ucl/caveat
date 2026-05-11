from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from caveat.data.augment import SequenceJitter
from caveat.encoding import BaseEncoder, LHS2RHSDataset


class Seq2ScoreEncoder(BaseEncoder):
    def __init__(
        self, max_length: int = 16, norm_duration: int = 2880, **kwargs
    ):
        self.max_length = max_length
        self.norm_duration = norm_duration
        self.jitter = kwargs.get("jitter", 0)

    def encode(
        self, schedules: pd.DataFrame, labels: Optional[Tensor]
    ) -> LHS2RHSDataset:
        # act encoding
        self.sos = 0
        self.eos = 1
        acts = set(schedules.act.unique())
        self.index_to_acts = {i + 2: a for i, a in enumerate(acts)}
        self.index_to_acts[0] = "<SOS>"
        self.index_to_acts[1] = "<EOS>"
        acts_to_index = {a: i for i, a in self.index_to_acts.items()}

        # mode encoding
        modes = set(schedules["mode"].unique())
        self.index_to_modes = {i: m for i, m in enumerate(modes)}
        self.modes_to_index = {m: i for i, m in self.index_to_modes.items()}

        self.act_encodings = len(self.index_to_acts)
        self.mode_encodings = len(self.index_to_modes)

        self.max_distance = schedules.distance.max()

        self.max_score = schedules.score.max()

        # prepare schedules dataframe
        schedules = schedules.copy()
        schedules.duration = schedules.duration / self.norm_duration
        schedules.act = schedules.act.map(acts_to_index)
        schedules["mode"] = schedules["mode"].map(self.modes_to_index)
        schedules["distance"] = schedules["distance"] / self.max_distance
        schedules["score"] = schedules["score"] / self.max_score

        # encode
        encoded_schedules, encoded_target, masks = self._encode_sequences(
            schedules, self.max_length
        )

        # augment
        augment = SequenceJitter(self.jitter) if self.jitter else None

        return LHS2RHSDataset(
            lhs=encoded_schedules,
            rhs=encoded_target,
            lhs_weights=masks,
            rhs_weights=masks,
            act_encodings=len(self.index_to_acts),
            mode_encodings=len(self.index_to_modes),
            activity_weights=None,
            augment=augment,
            labels=labels,
        )

    def _encode_sequences(
        self, data: pd.DataFrame, max_length: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # calc weightings
        act_weights = self._calc_act_weights(data)
        # act_weights = self._unit_act_weights(self.encodings)

        persons = data.pid.nunique()
        encoding_width = 4  # act, duration, mode, distance

        encoded = np.zeros(
            (persons, max_length, encoding_width), dtype=np.float32
        )
        target = np.zeros((persons), dtype=np.float32)
        weights = np.zeros((persons, max_length), dtype=np.float32)

        for pid, (_, trace) in enumerate(data.groupby("pid")):
            score = trace.score.iloc[0]
            seq_encoding, seq_weights = encode_sequences(
                acts=list(trace.act),
                durations=list(trace.duration),
                modes=list(trace["mode"]),
                distances=list(trace["distance"]),
                max_length=max_length,
                encoding_width=encoding_width,
                act_weights=act_weights,
                sos=self.sos,
                eos=self.eos,
            )
            encoded[pid] = seq_encoding  # [N, L, W]
            target[pid] = score  # [N]
            weights[pid] = seq_weights  # [N, L]

        return (
            torch.from_numpy(encoded),
            torch.from_numpy(target),
            torch.from_numpy(weights),
        )

    def _calc_act_weights(self, data: pd.DataFrame) -> Dict[str, float]:
        act_weights = (
            data.groupby("act", observed=True).duration.sum().to_dict()
        )
        n = data.pid.nunique()
        act_weights.update({self.sos: n, self.eos: n})
        act_weights = np.array(
            [act_weights[k] for k in range(len(act_weights))]
        )
        act_weights = 1 / act_weights
        return act_weights

    def _unit_act_weights(self, n: int) -> Dict[str, float]:
        return np.array([1 for _ in range(n)])

    def decode_input(self, schedules: Tensor) -> pd.DataFrame:
        return self.to_dataframe(schedules)

    def decode_target(self, schedules: Tensor) -> pd.DataFrame:
        return pd.DataFrame(schedules.numpy(), columns=["score"])

    def decode_output(self, schedules: Tensor) -> pd.DataFrame:
        return pd.DataFrame(schedules.numpy(), columns=["score"])

    def decode(self, schedules: Tensor) -> pd.DataFrame:
        """Decode a sequences ([N, max_length, encoding]) into DataFrame of 'traces', eg:

        pid | act | start | end

        enumeration of seq is used for pid.

        Args:
            schedules (Tensor): _description_

        Returns:
            pd.DataFrame: _description_
        """
        schedules = self.pack(schedules)
        return self.to_dataframe(schedules)

    def pack(self, schedules: Tensor) -> Tensor:
        schedules, durations, modes, distances = torch.split(
            schedules, [self.act_encodings, 1, self.mode_encodings, 1], dim=-1
        )
        schedules = schedules.argmax(dim=-1).unsqueeze(-1)
        modes = modes.argmax(dim=-1).unsqueeze(-1)
        return torch.cat([schedules, durations, modes, distances], dim=-1)

    def to_dataframe(self, schedules: Tensor):
        schedules, durations, modes, distances = torch.split(
            schedules, [1, 1, 1, 1], dim=-1
        )
        decoded = []
        for pid in range(len(schedules)):
            act_start = 0
            for act_idx, duration, mode_idx, distance in zip(
                schedules[pid], durations[pid], modes[pid], distances[pid]
            ):
                if int(act_idx) == self.sos:
                    continue
                if int(act_idx) == self.eos:
                    break
                duration = int(duration * self.norm_duration)
                decoded.append(
                    [
                        pid,
                        self.index_to_acts[int(act_idx)],
                        act_start,
                        act_start + duration,
                        self.index_to_modes[int(mode_idx)],
                        float(distance * self.max_distance),
                    ]
                )
                act_start += duration

        df = pd.DataFrame(
            decoded, columns=["pid", "act", "start", "end", "mode", "distance"]
        )
        df["duration"] = df.end - df.start
        return df


def encode_sequences(
    acts: list[int],
    durations: list[float],
    modes: list[int],
    distances: list[float],
    max_length: int,
    encoding_width: int,
    act_weights: np.ndarray,
    sos: int,
    eos: int,
) -> Tuple[np.ndarray, np.ndarray]:
    encoding = np.zeros((max_length, encoding_width), dtype=np.float32)
    weights = np.zeros((max_length), dtype=np.float32)
    # SOS
    encoding[0][0] = sos
    # mask includes sos
    weights[0] = act_weights[sos]

    for i in range(1, max_length):
        if i < len(acts) + 1:
            encoding[i][0] = acts[i - 1]
            encoding[i][1] = durations[i - 1]
            encoding[i][2] = modes[i - 1]
            encoding[i][3] = distances[i - 1]
            weights[i] = act_weights[acts[i - 1]]
        elif i < len(acts) + 2:
            encoding[i][0] = eos
            # mask includes first eos
            weights[i] = act_weights[eos]
        else:
            encoding[i][0] = eos
            # act weights are 0 for padding eos

    return encoding, weights
