from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from caveat.data.augment import SequenceJitter
from caveat.encoding import (
    BaseDataset,
    BaseEncoder,
    StaggeredDataset,
    act_weight_library,
    seq_weight_library,
)


class SequenceEncoder(BaseEncoder):
    dataset = BaseDataset

    def __init__(
        self, max_length: int = 12, norm_duration: int = 1440, **kwargs
    ):
        """Sequence Encoder for sequences of activities. Also supports conditional attributes.

        Args:
            max_length (int, optional): _description_. Defaults to 12.
            norm_duration (int, optional): _description_. Defaults to 1440.
        """
        self.max_length = max_length
        self.norm_duration = norm_duration
        self.jitter = kwargs.get("jitter", 0)
        self.fix_durations = kwargs.get("fix_durations", False)
        self.encodings = None  # initialise as none so we can check for encoding versus re-encoding
        self.weighting = kwargs.get("weighting", "unit")
        self.joint_weighting = kwargs.get("joint_weighting", "unit")
        self.trim_eos = kwargs.get("trim_eos", True)

        self.weighter = act_weight_library.get(self.weighting, None)
        if self.weighter is None:
            raise ValueError(
                f"Unknown Sequence Encoder weighting: {self.weighting}, should be one of: {act_weight_library.keys()}"
            )

        self.joint_weighter = seq_weight_library.get(self.joint_weighting, None)
        if self.joint_weighter is None:
            raise ValueError(
                f"Unknown Sequence Encoder weighting: {self.joint_weighting}, should be one of: {seq_weight_library.keys()}"
            )

        print(
            f"""Sequence Encoder initialised with:
        max_length: {self.max_length}
        norm_duration: {self.norm_duration}
        jitter: {self.jitter}
        fix_durations: {self.fix_durations}
        (act) weighting: {self.weighting}
        (seq) joint weighting: {self.joint_weighting}
        trim eos: {self.trim_eos}
        """
        )

    def encode(
        self,
        schedules: pd.DataFrame,
        labels: Optional[Tensor],
        label_weights: Optional[Tensor],
    ) -> BaseDataset:
        if labels is not None:
            assert schedules.pid.nunique() == labels.shape[0]
        if self.encodings is None:
            self.setup_encoder(schedules)
        return self._encode(schedules, labels, label_weights)

    def setup_encoder(self, schedules: pd.DataFrame) -> None:
        self.sos = 0
        self.eos = 1

        self.index_to_acts = {
            int(i + 2): a for i, a in enumerate(schedules.act.unique())
        }
        self.index_to_acts[0] = "<SOS>"
        self.index_to_acts[1] = "<EOS>"
        self.acts_to_index = {a: i for i, a in self.index_to_acts.items()}

        self.encodings = len(self.index_to_acts)

    def _encode(
        self,
        schedules: pd.DataFrame,
        labels: Optional[Tensor],
        label_weights: Optional[Tuple[Tensor, Tensor]],
    ) -> dataset:
        # prepare schedules dataframe
        schedules = schedules.copy()
        schedules.duration = schedules.duration / self.norm_duration
        schedules.act = schedules.act.map(self.acts_to_index).astype(int)

        # encode
        encoded_schedules = self._encode_sequences(schedules, self.max_length)

        # weights
        act_weights = self.weighter(
            sequences=encoded_schedules,
            sos_idx=self.sos,
            eos_idx=self.eos,
            trim_eos=self.trim_eos,
        )
        seq_weights = self.joint_weighter(sequences=encoded_schedules)

        # unpack label weights
        label_weights, joint_weights = label_weights

        # augment
        augment = SequenceJitter(self.jitter) if self.jitter else None

        return self.dataset(
            schedules=encoded_schedules,
            act_weights=act_weights,
            seq_weights=seq_weights,
            activity_encodings=len(self.index_to_acts),
            activity_weights=None,
            augment=augment,
            labels=labels,
            label_weights=label_weights,
            joint_weights=joint_weights,
        )

    def _encode_sequences(
        self, data: pd.DataFrame, max_length: int
    ) -> Tuple[Tensor, Tensor]:

        persons = data.pid.nunique()
        encoding_width = 2  # cat act encoding plus duration

        encoded = np.zeros(
            (persons, max_length, encoding_width), dtype=np.float32
        )

        for pid, (_, trace) in enumerate(data.groupby("pid")):
            seq_encoding = encode_sequence(
                acts=list(trace.act),
                durations=list(trace.duration),
                max_length=max_length,
                encoding_width=encoding_width,
                sos=self.sos,
                eos=self.eos,
            )
            encoded[pid] = seq_encoding  # [N, L, W]

        return torch.from_numpy(encoded)

    def decode(self, schedules: Tensor, argmax=True) -> pd.DataFrame:
        """Decode a sequences ([N, max_length, encoding]) into DataFrame of 'traces', eg:

        pid | act | start | end

        enumeration of seq is used for pid.

        Args:
            schedules (Tensor): _description_

        Returns:
            pd.DataFrame: _description_
        """
        if argmax:
            schedules, durations = torch.split(
                schedules, [self.encodings, 1], dim=-1
            )
            schedules = schedules.argmax(dim=-1).numpy()
        else:
            schedules, durations = torch.split(schedules, [1, 1], dim=-1)

        decoded = []

        for pid in range(len(schedules)):
            act_start = 0
            for act_idx, duration in zip(schedules[pid], durations[pid]):
                if int(act_idx) == self.sos:
                    continue
                if int(act_idx) == self.eos:
                    if act_start == 0:
                        print(f"Failed to decode pid: {pid}")
                        decoded.append(
                            [pid, "home", 0, 0]
                        )  # todo: hack for empty plan
                    break
                # denormalise incrementally preserves duration
                duration = int(duration * self.norm_duration)
                decoded.append(
                    [
                        pid,
                        self.index_to_acts[int(act_idx)],
                        act_start,
                        act_start + duration,
                    ]
                )
                act_start += duration

        df = pd.DataFrame(decoded, columns=["pid", "act", "start", "end"])
        df["duration"] = df.end - df.start

        if self.fix_durations:
            # ensure durations sum to norm duration
            df = norm_durations(df, self.norm_duration)
            # ensure last end time is norm duration
            df = fix_end_durations(df, self.norm_duration)
            df["duration"] = df.end - df.start

        return df


class SequenceEncoderStaggered(SequenceEncoder):
    dataset = StaggeredDataset


def encode_sequence(
    acts: list[int],
    durations: list[float],
    max_length: int,
    encoding_width: int,
    sos: int,
    eos: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequence encoding from ranges.

    Args:
        acts (Iterable[int]): _description_
        durations (Iterable[float]): _description_
        max_length (int): _description_
        encoding_width (dict): _description_
        sos (int): _description_
        eos (int): _description_

    Returns:
        np.ndarray: _description_
    """
    encoding = np.zeros((max_length, encoding_width), dtype=np.float32)
    encoding[0][0] = sos

    for i in range(1, max_length):
        if i < len(acts) + 1:
            encoding[i][0] = acts[i - 1]
            encoding[i][1] = durations[i - 1]
        elif i < len(acts) + 2:
            encoding[i][0] = eos
        else:
            encoding[i][0] = eos

    return encoding


def fix_end_durations(data: pd.DataFrame, end_duration: int) -> pd.DataFrame:
    data.loc[data.groupby(data.pid).tail(1).index, "end"] = end_duration
    return data


def norm_durations(data: pd.DataFrame, target_duration: int) -> pd.DataFrame:
    def norm_plan_durations(plan: pd.DataFrame):
        plan_duration = plan.duration.sum()
        if plan_duration == 0:
            print("Zero duration plan found, cannot normalise")
            return plan
        if plan_duration != target_duration:
            r = target_duration / plan_duration
            plan.duration = (plan.duration * r).astype(int)
            accumulated = list(plan.duration.cumsum())
            plan.start = [0] + accumulated[:-1]
            plan.end = accumulated
        return plan.reset_index(drop=True)

    data = (
        data.groupby(data.pid).apply(norm_plan_durations).reset_index(drop=True)
    )
    return data
