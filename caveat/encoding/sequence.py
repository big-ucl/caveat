from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from caveat.data.augment import SequenceJitter
from caveat.encoding import BaseDataset, BaseEncoder, StaggeredDataset


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
        label_weights: Optional[Tensor],
    ) -> dataset:
        # prepare schedules dataframe
        schedules = schedules.copy()
        schedules.duration = schedules.duration / self.norm_duration
        schedules.act = schedules.act.map(self.acts_to_index).astype(int)

        # encode
        encoded_schedules, masks = self._encode_sequences(
            schedules, self.max_length
        )

        # augment
        augment = SequenceJitter(self.jitter) if self.jitter else None

        return self.dataset(
            schedules=encoded_schedules,
            schedule_weights=masks,
            activity_encodings=len(self.index_to_acts),
            activity_weights=None,
            augment=augment,
            labels=labels,
            label_weights=label_weights,
        )

    def _encode_sequences(
        self, data: pd.DataFrame, max_length: int
    ) -> Tuple[Tensor, Tensor]:

        # calc weightings
        act_weights = self._calc_act_weights(data)
        # act_weights = self._unit_act_weights(self.encodings)

        persons = data.pid.nunique()
        encoding_width = 2  # cat act encoding plus duration

        encoded = np.zeros(
            (persons, max_length, encoding_width), dtype=np.float32
        )
        weights = np.zeros((persons, max_length), dtype=np.float32)

        for pid, (_, trace) in enumerate(data.groupby("pid")):
            seq_encoding, seq_weights = encode_sequence(
                acts=list(trace.act),
                durations=list(trace.duration),
                max_length=max_length,
                encoding_width=encoding_width,
                act_weights=act_weights,
                sos=self.sos,
                eos=self.eos,
            )
            encoded[pid] = seq_encoding  # [N, L, W]
            weights[pid] = seq_weights  # [N, L]

        return (torch.from_numpy(encoded), torch.from_numpy(weights))

    def _calc_act_weights(self, data: pd.DataFrame) -> Dict[str, float]:

        act_weights = data.act.value_counts().to_dict()
        n = data.pid.nunique()
        act_weights.update({self.sos: n, self.eos: n})
        for act in self.index_to_acts.keys():
            if act not in act_weights:
                act_weights[act] = n
        act_weights = np.array(
            [act_weights[k] for k in range(len(act_weights))]
        )
        act_weights = 1 / act_weights
        return act_weights

    def _unit_act_weights(self, n: int) -> Dict[str, float]:
        return np.array([1 for _ in range(n)])

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
    act_weights: np.ndarray,
    sos: int,
    eos: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequence encoding from ranges.

    Args:
        acts (Iterable[int]): _description_
        durations (Iterable[float]): _description_
        max_length (int): _description_
        encoding_width (dict): _description_
        act_weights (dict): _description_
        sos (int): _description_
        eos (int): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
    """
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
            weights[i] = act_weights[acts[i - 1]]
        elif i < len(acts) + 2:
            encoding[i][0] = eos
            # mask includes first eos
            weights[i] = act_weights[eos]
        else:
            encoding[i][0] = eos
            # act weights are 0 for padding eos

    return encoding, weights


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
