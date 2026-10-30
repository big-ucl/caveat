from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from caveat.data.augment import SequenceJitter
from caveat.encoding import BaseDataset, BaseEncoder, StaggeredDataset
from caveat.encoding.cont_weighting import (
    act_weight_library,
    seq_weight_library,
)


class ContinuousEncoder(BaseEncoder):
    dataset = BaseDataset

    def __init__(
        self, max_length: int = 12, norm_duration: int = 1440, **kwargs
    ):
        """Continuous Encoder for sequences of activities. Also supports conditional attributes.

        Args:
            max_length (int, optional): _description_. Defaults to 12.
            norm_duration (int, optional): _description_. Defaults to 1440.
        """
        self.max_length = max_length
        self.norm_duration = norm_duration
        self.jitter = kwargs.get("jitter", 0)

        fixes = ["none", "stretch", "crop"]
        self.fix_durations = kwargs.get("fix_durations", "stretch")
        if self.fix_durations not in fixes:
            raise ValueError(
                f"Unknown fix_durations option: {self.fix_durations}, should be one of: {fixes}"
            )

        self.encodings = None  # initialise as none so we can check for encoding versus re-encoding
        self.weighting = kwargs.get("weighting", "unit")
        self.joint_weighting = kwargs.get("joint_weighting", "unit")
        self.trim_eos = kwargs.get("trim_eos", True)

        self.weighter = act_weight_library.get(self.weighting, None)
        if self.weighter is None:
            raise ValueError(
                f"Unknown Continuous Encoder weighting: {self.weighting}, should be one of: {act_weight_library.keys()}"
            )

        self.joint_weighter = seq_weight_library.get(self.joint_weighting, None)
        if self.joint_weighter is None:
            raise ValueError(
                f"Unknown Continuous Encoder weighting: {self.joint_weighting}, should be one of: {seq_weight_library.keys()}"
            )

        print(
            f"""Continuous Encoder initialised with:
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
        schedules = pd.DataFrame(
            {
                "pid": schedules.pid.to_numpy(),
                "act": schedules.act.map(self.acts_to_index)
                .astype(np.int64)
                .to_numpy(),
                "duration": schedules.duration.to_numpy() / self.norm_duration,
            }
        )

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
        if label_weights is None:
            label_weights = (None, None)
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

    def _slow_encode_sequences(
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

    def slow_decode(
        self, schedules: Tensor, argmax=True, chunk_size: int | None = None
    ) -> pd.DataFrame:
        """Decode a sequences ([N, max_length, encoding]) into DataFrame of 'traces', eg:

        pid | act | start | end

        enumeration of seq is used for pid.
        """
        with torch.no_grad():
            if argmax:
                acts, durations = torch.split(
                    schedules, [self.encodings, 1], dim=-1
                )
                acts = acts.argmax(dim=-1)
            else:
                acts, durations = torch.split(schedules, [1, 1], dim=-1)
                acts = acts.squeeze(-1)
            durations = durations.squeeze(-1)

            # single detach + move to cpu, no lingering graph, no GPU tensor held around
            acts = acts.detach().to("cpu", non_blocking=False).numpy()
            durations = durations.detach().to("cpu", non_blocking=False).numpy()

        del schedules  # drop reference to the (potentially large) input tensor ASAP

        N = acts.shape[0]
        step = (
            chunk_size or N
        )  # process everything at once unless a chunk_size is given

        decoded = []
        for start in range(0, N, step):
            end = min(start + step, N)
            for pid in range(start, end):
                act_start = 0
                for act_idx, duration in zip(acts[pid], durations[pid]):
                    act_idx = int(act_idx)
                    if act_idx == self.sos:
                        continue
                    if act_idx == self.eos:
                        if act_start == 0:
                            print(
                                f"Failed to decode pid: {pid} returning empty plan"
                            )
                            decoded.append(
                                (pid, "home", 0, 0)
                            )  # todo: hack for empty plan
                        break
                    dur = int(duration * self.norm_duration)
                    decoded.append(
                        (
                            pid,
                            self.index_to_acts[act_idx],
                            act_start,
                            act_start + dur,
                        )
                    )
                    act_start += dur

        df = pd.DataFrame(decoded, columns=["pid", "act", "start", "end"])
        df["duration"] = df.end - df.start

        if self.fix_durations == "none":
            return df
        if self.fix_durations == "stretch":
            df = norm_durations(df, self.norm_duration)
            df = fix_end_durations(df, self.norm_duration)
            df["duration"] = df.end - df.start
            return df
        if self.fix_durations == "crop":
            df = df[df.end <= self.norm_duration]
            df = fix_end_durations(df, self.norm_duration)
            df["duration"] = df.end - df.start
            return df

        raise ValueError(f"Unknown fix_durations option: {self.fix_durations}")

    def decode(
        self, schedules: Tensor, argmax=True, chunk_size: int | None = None
    ) -> pd.DataFrame:
        """Decode a sequences ([N, max_length, encoding]) into DataFrame of 'traces', eg:

        pid | act | start | end

        enumeration of seq is used for pid.
        """
        with torch.no_grad():
            if argmax:
                acts, durations = torch.split(
                    schedules, [self.encodings, 1], dim=-1
                )
                acts = acts.argmax(dim=-1)
            else:
                acts, durations = torch.split(schedules, [1, 1], dim=-1)
                acts = acts.squeeze(-1)
            durations = durations.squeeze(-1)

            acts = acts.detach().to("cpu", non_blocking=False).numpy()
            durations = durations.detach().to("cpu", non_blocking=False).numpy()

        del schedules

        # index -> name lookup (indices are contiguous 0..encodings-1)
        names = np.empty(self.encodings, dtype=object)
        for i, a in self.index_to_acts.items():
            names[i] = a

        N = acts.shape[0]
        step = chunk_size or N  # chunking now only bounds temporary memory

        frames = [
            self._decode_chunk(
                acts[s : s + step], durations[s : s + step], s, names
            )
            for s in range(0, N, step)
        ]
        df = pd.concat(frames, ignore_index=True)
        df["duration"] = df.end - df.start

        if self.fix_durations == "none":
            return df
        if self.fix_durations == "stretch":
            df = norm_durations(df, self.norm_duration)
            df = fix_end_durations(df, self.norm_duration)
            df["duration"] = df.end - df.start
            return df
        if self.fix_durations == "crop":
            df = df[df.end <= self.norm_duration]
            df = fix_end_durations(df, self.norm_duration)
            df["duration"] = df.end - df.start
            return df

        raise ValueError(f"Unknown fix_durations option: {self.fix_durations}")

    def _decode_chunk(
        self,
        acts: np.ndarray,
        durations: np.ndarray,
        offset: int,
        names: np.ndarray,
    ) -> pd.DataFrame:
        acts = acts.astype(np.int64)  # truncates like int() if argmax=False

        is_eos = acts == self.eos
        eos_seen = np.cumsum(is_eos, axis=1)  # 1 at and after the first EOS
        valid = (eos_seen == 0) & (acts != self.sos)  # real activities only

        dur = np.where(
            valid, (durations * self.norm_duration).astype(np.int64), 0
        )
        ends = np.cumsum(dur, axis=1)
        starts = ends - dur

        # empty plan hack: hit an EOS having accumulated no time
        empty = is_eos.any(axis=1) & (dur.sum(axis=1) == 0)
        if empty.any():
            print(
                f"Failed to decode {int(empty.sum())} plans in chunk, returning empty plan"
            )
        home = is_eos & (eos_seen == 1) & empty[:, None]

        rows, cols = np.nonzero(valid | home)  # row-major: pid then slot
        act_names = names[acts[rows, cols]]
        act_names[home[rows, cols]] = "home"

        return pd.DataFrame(
            {
                "pid": rows + offset,
                "act": act_names,
                "start": starts[rows, cols],
                "end": ends[rows, cols],
            }
        )

    def _encode_sequences(self, data: pd.DataFrame, max_length: int) -> Tensor:
        encoding_width = 2  # cat act encoding plus duration

        # groupby sorts by pid, so a stable sort reproduces the same person order
        data = data.sort_values("pid", kind="stable")
        pid_codes, uniques = pd.factorize(data.pid.to_numpy(), sort=True)
        n = len(uniques)

        acts = data.act.to_numpy()
        durs = data.duration.to_numpy(dtype=np.float32)

        # 1-based position of each activity within its person (slot 0 is SOS)
        counts = np.bincount(pid_codes, minlength=n)
        starts = np.cumsum(counts) - counts
        pos = np.arange(len(data)) - starts[pid_codes] + 1

        encoded = np.zeros((n, max_length, encoding_width), dtype=np.float32)
        encoded[:, 0, 0] = self.sos
        encoded[:, 1:, 0] = self.eos  # EOS marker and padding, duration stays 0

        # truncate to max_length - 1 activities, same as the loop
        keep = pos < max_length
        rows, cols = pid_codes[keep], pos[keep]
        encoded[rows, cols, 0] = acts[keep]
        encoded[rows, cols, 1] = durs[keep]

        return torch.from_numpy(encoded)


class ContinuousEncoderStaggered(ContinuousEncoder):
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
    print(f"Normalising plan durations to {target_duration} minutes")
    zero_duration_counter = 0

    def norm_plan_durations(plan: pd.DataFrame, zero_duration_counter):
        plan_duration = plan.duration.sum()
        if plan_duration == 0:
            zero_duration_counter += 1
            return plan
        if plan_duration != target_duration:
            r = target_duration / plan_duration
            plan.duration = (plan.duration * r).astype(int)
            accumulated = list(plan.duration.cumsum())
            plan.start = [0] + accumulated[:-1]
            plan.end = accumulated
        return plan.reset_index(drop=True)

    data = (
        data.groupby(data.pid)
        .apply(lambda x: norm_plan_durations(x, zero_duration_counter))
        .reset_index(drop=True)
    )

    print(
        f"Normalised {len(data.pid.unique())} plans, {zero_duration_counter} had zero duration and were not normalised"
    )

    return data
