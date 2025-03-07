from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from caveat.data.augment import DiscreteJitter
from caveat.encoding import BaseDataset, BaseEncoder, PaddedDatatset


class DiscreteEncoder(BaseEncoder):
    def __init__(self, duration: int = 1440, step_size: int = 10, **kwargs):
        self.duration = duration
        self.step_size = step_size
        self.steps = duration // step_size
        self.jitter = kwargs.get("jitter", 0)
        self.acts_to_index = None
        print(
            f"DiscreteEncoder: {self.duration=}, {self.step_size=}, {self.jitter=}"
        )

    def encode(
        self,
        schedules: pd.DataFrame,
        labels: Optional[Tensor],
        label_weights: Optional[Tensor],
    ) -> BaseDataset:
        if self.acts_to_index is None:
            self.setup_encoder(schedules)
        return self._encode(schedules, labels, label_weights)

    def setup_encoder(self, schedules: pd.DataFrame) -> None:
        self.index_to_acts = {
            i: a for i, a in enumerate(schedules.act.unique())
        }
        self.acts_to_index = {a: i for i, a in self.index_to_acts.items()}

        # calc weightings
        act_freqs = (
            schedules.groupby("act", observed=True).duration.sum().to_dict()
        )
        index_freqs = {self.acts_to_index[k]: v for k, v in act_freqs.items()}
        ordered_freqs = np.array(
            [index_freqs[k] for k in range(len(index_freqs))]
        )
        weights = 1 / np.log(ordered_freqs)
        weights = (
            weights / weights.mean()
        )  # normalise to average 1 for each activity
        weights = (
            weights / self.steps
        )  # normalise to average 1 for each activity schedule
        self.encoding_weights = torch.from_numpy(weights).float()

    def _encode(
        self,
        schedules: pd.DataFrame,
        labels: Optional[Tensor],
        label_weights: Optional[Tuple[Tensor, Tensor]],
    ) -> BaseDataset:
        if label_weights is None:
            label_weights = (None, None)
        label_weights, joint_weights = label_weights

        schedules = schedules.copy()
        schedules.act = schedules.act.map(self.acts_to_index)
        activity_encodings = schedules.act.nunique()

        encoded = discretise_population(
            schedules, duration=self.duration, step_size=self.step_size
        )

        augment = (
            DiscreteJitter(self.step_size, self.jitter) if self.jitter else None
        )

        return BaseDataset(
            schedules=encoded.long(),
            act_weights=torch.ones(encoded.shape),
            seq_weights=torch.ones(encoded.shape[0], 1),
            activity_encodings=activity_encodings,
            activity_weights=self.encoding_weights,
            augment=augment,
            labels=labels,
            label_weights=label_weights,
            joint_weights=joint_weights,
        )

    def decode(self, schedules: Tensor, argmax=True) -> pd.DataFrame:
        """Decode decretised a sequences ([B, C, T, A]) into DataFrame of 'traces', eg:

        pid | act | start | end

        pid is taken as sample enumeration.

        Args:
            encoded (Tensor): _description_
            mapping (dict): _description_
            length (int): Length of plan in minutes.

        Returns:
            pd.DataFrame: _description_
        """
        if argmax:
            schedules = torch.argmax(schedules, dim=-1)
        decoded = []

        for pid in range(len(schedules)):
            current_act = None
            act_start = 0

            for step, act_idx in enumerate(schedules[pid]):
                if int(act_idx) != current_act and current_act is not None:
                    decoded.append(
                        [
                            pid,
                            self.index_to_acts[current_act],
                            int(act_start * self.step_size),
                            int(step * self.step_size),
                        ]
                    )
                    act_start = step
                current_act = int(act_idx)
            decoded.append(
                [
                    pid,
                    self.index_to_acts[current_act],
                    int(act_start * self.step_size),
                    self.duration,
                ]
            )

        return pd.DataFrame(decoded, columns=["pid", "act", "start", "end"])


class DiscreteEncoderPadded(BaseEncoder):
    def __init__(self, duration: int = 1440, step_size: int = 10, **kwargs):
        self.duration = duration
        self.step_size = step_size
        self.steps = duration // step_size
        self.jitter = kwargs.get("jitter", 0)
        print(
            f"DiscreteEncoderPadded: {self.duration=}, {self.step_size=}, {self.jitter=}"
        )

    def encode(
        self, schedules: pd.DataFrame, conditionals: Optional[Tensor]
    ) -> PaddedDatatset:
        self.index_to_acts = {
            i + 1: a for i, a in enumerate(schedules.act.unique())
        }
        self.index_to_acts[0] = "<PAD>"
        acts_to_index = {a: i for i, a in self.index_to_acts.items()}

        schedules = schedules.copy()
        schedules.act = schedules.act.map(acts_to_index)
        activity_encodings = len(acts_to_index)

        # calc weightings
        weights = (
            schedules.groupby("act", observed=True).duration.sum().to_dict()
        )
        weights[0] = (
            schedules.pid.nunique() * 60
        )  # pad weight is equal to 1 hour
        weights = np.array([weights[k] for k in range(len(weights))])
        activity_weights = torch.from_numpy(1 / (weights)).float()
        encoded = discretise_population(
            schedules, duration=self.duration, step_size=self.step_size
        )
        masks = torch.ones(
            (encoded.shape[0], encoded.shape[-1] + 1), dtype=torch.int8
        )

        augment = (
            DiscreteJitter(self.step_size, self.jitter) if self.jitter else None
        )

        return PaddedDatatset(
            schedules=encoded.long(),
            act_weights=masks,
            seq_weights=torch.ones(encoded.shape[0], 1),
            activity_encodings=activity_encodings,
            activity_weights=activity_weights,
            augment=augment,
            labels=conditionals,
            label_weights=None,
            joint_weights=None,
        )

    def decode(self, schedules: Tensor) -> pd.DataFrame:
        """Decode disretised a sequences ([B, C, T, A]) into DataFrame of 'traces', eg:

        pid | act | start | end

        pid is taken as sample enumeration.

        Args:
            encoded (Tensor): _description_
            mapping (dict): _description_
            length (int): Length of plan in minutes.

        Returns:
            pd.DataFrame: _description_
        """
        schedules = torch.argmax(schedules, dim=-1)
        decoded = []

        for pid in range(len(schedules)):
            current_act = None
            act_start = 0

            for step, act_idx in enumerate(schedules[pid]):
                if int(act_idx) != current_act and current_act is not None:
                    decoded.append(
                        [
                            pid,
                            self.index_to_acts[current_act],
                            int(act_start * self.step_size),
                            int(step * self.step_size),
                        ]
                    )
                    act_start = step
                current_act = int(act_idx)

        return pd.DataFrame(decoded, columns=["pid", "act", "start", "end"])


def discretise_population(
    data: pd.DataFrame, duration: int, step_size: int
) -> torch.Tensor:
    """Convert given population of activity traces into vector [N, L] of classes.
    N is the population size.
    L is time steps.

    Args:
        data (pd.DataFrame): _description_
        duration (int): _description_
        step_size (int): _description_

    Returns:
        torch.tensor: [N, L]
    """
    persons = data.pid.nunique()
    steps = duration // step_size
    encoded = np.zeros((persons, steps))

    for pid, (_, trace) in enumerate(data.groupby("pid")):
        trace_encoding = discretise_trace(
            acts=trace.act, starts=trace.start, ends=trace.end, length=duration
        )
        trace_encoding = down_sample(trace_encoding, step_size)
        encoded[pid] = trace_encoding  # [N, L]
    return torch.from_numpy(encoded)


def discretise_trace(
    acts: Iterable[int], starts: Iterable[int], ends: Iterable[int], length: int
) -> np.ndarray:
    """Create categorical encoding from ranges with step of 1.

    Args:
        acts (Iterable[str]): _description_
        starts (Iterable[int]): _description_
        ends (Iterable[int]): _description_
        length (int): _description_

    Returns:
        np.array: _description_
    """
    encoding = np.zeros((length))
    for act, start, end in zip(acts, starts, ends):
        encoding[start:end] = act
    return encoding


def down_sample(array: np.ndarray, step: int) -> np.ndarray:
    """Down-sample by steppiong through given array.
    todo:
    Methodology will down sample based on first classification.
    If we are down sampling a lot (for example from minutes to hours),
    we would be better of, sampling based on majority class.

    Args:
        array (np.array): _description_
        step (int): _description_

    Returns:
        np.array: _description_
    """
    return array[::step]
