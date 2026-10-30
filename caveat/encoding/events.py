from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from caveat.encoding.continuous import (
    ContinuousEncoder,
    fix_end_durations,
    norm_durations,
)


class EventEncoder(ContinuousEncoder):
    """Event encoder: each activity is one token (type, start_bin, end_bin).

    Follows the 'Deep Activity Model' representation. Times are discretised into
    `n_bins` slots (96 x 15min for a 1440min day) and predicted as classifications.

    Encoded schedule layout, shape [N, max_length, 3] (float32, like the continuous
    encoder, cast to long in the model):

        [:, :, 0]  activity index (0=<SOS> unused in the sequence, 1=<EOS>, 2+ = acts)
        [:, :, 1]  start bin in [0, n_bins]
        [:, :, 2]  end bin in [0, n_bins]

    Events start at slot 0 (there is no <SOS> slot, the model supplies its own start
    token). After the last event comes <EOS>, then <EOS> padding. EOS/pad slots have
    start=end=0 and are masked out of the time losses.
    """

    def __init__(
        self,
        max_length: int = 12,
        norm_duration: int = 1440,
        n_bins: int = 96,
        **kwargs,
    ):
        super().__init__(
            max_length=max_length, norm_duration=norm_duration, **kwargs
        )
        # sequence weighting: None | "inverse_seq" (1/freq of the activity-type
        # sequence, times ignored), tempered by weight_power, optionally clipped
        self.event_weighting = kwargs.get("event_weighting", "none")
        if self.event_weighting not in ("none", "inverse_seq"):
            raise ValueError(
                f"Unknown event_weighting: {self.event_weighting}, should be None or 'inverse_seq'"
            )
        self.weight_power = kwargs.get("weight_power", 1.0)
        self.max_weight = kwargs.get("max_weight", None)
        self.min_count = kwargs.get("min_count", 1)
        self.max_weight = kwargs.get("max_weight", None)
        self.n_bins = n_bins
        self.time_classes = (
            n_bins + 1
        )  # bins 0..n_bins inclusive (end of day = n_bins)
        self.bin_size = norm_duration / n_bins
        if self.jitter:
            print(
                "EventEncoder: jitter is not supported for binned times, disabling."
            )
            self.jitter = 0
        print(
            f"""Event Encoder:
        n_bins: {self.n_bins} (bin size {self.bin_size} min)
        time classes: {self.time_classes}
        """
        )

    # ------------------------------------------------------------------ encode
    def _encode(
        self,
        schedules: pd.DataFrame,
        labels: Optional[Tensor],
        label_weights: Optional[Tuple[Tensor, Tensor]],
    ):
        schedules = pd.DataFrame(
            {
                "pid": schedules.pid.to_numpy(),
                "act": schedules.act.map(self.acts_to_index)
                .astype(np.int64)
                .to_numpy(),
                "duration": schedules.duration.to_numpy(),  # minutes
            }
        )

        encoded_schedules = self._encode_sequences(schedules, self.max_length)

        # NOTE: assumes the weighters read the act index from [:, :, 0] and don't
        # care about trailing width (I haven't seen act_weight_library).
        act_weights = self.weighter(
            sequences=encoded_schedules,
            sos_idx=self.sos,
            eos_idx=self.eos,
            trim_eos=self.trim_eos,
        )
        seq_weights = self.joint_weighter(sequences=encoded_schedules)
        if self.event_weighting == "inverse_seq":
            seq_weights = self._inverse_seq_weights(encoded_schedules)

        else:
            label_weights = (None, None)
        label_weights, joint_weights = label_weights

        return self.dataset(
            schedules=encoded_schedules,
            act_weights=act_weights,
            seq_weights=seq_weights,
            activity_encodings=len(self.index_to_acts),
            activity_weights=None,
            augment=None,
            labels=labels,
            label_weights=label_weights,
            joint_weights=joint_weights,
        )

    def _inverse_seq_weights(self, encoded: Tensor) -> Tensor:
        """Per-person weight = (1 / max(freq, min_count)) ** power, where freq is
        the count of the person's activity-type sequence. Normalised to mean 1 (optionally clipped at max_weight, then renormalised).
        Returns float32 [N], in the same person order as the encoded schedules.
        """
        types = encoded[:, :, 0].numpy().astype(np.int64)
        _, inv, counts = np.unique(
            types, axis=0, return_inverse=True, return_counts=True
        )
        freq = counts[inv.reshape(-1)].astype(np.float64)
        w = (1.0 / np.maximum(freq, self.min_count)) ** self.weight_power
        w = w / w.mean()
        if self.max_weight is not None:
            w = np.minimum(w, self.max_weight)
            w = w / w.mean()
        print(
            f"Inverse seq weighting: {len(counts)} unique type-sequences, "
            f"most common = {counts.max() / len(types):.1%} of people, "
            f"weights min/median/max = {w.min():.3g}/{np.median(w):.3g}/{w.max():.3g}"
        )
        return torch.from_numpy(w.astype(np.float32))

    def _encode_sequences(self, data: pd.DataFrame, max_length: int) -> Tensor:
        data = data.sort_values("pid", kind="stable")
        pid_codes, uniques = pd.factorize(data.pid.to_numpy(), sort=True)
        n = len(uniques)

        acts = data.act.to_numpy()
        durs = data.duration.to_numpy(dtype=np.float64)

        counts = np.bincount(pid_codes, minlength=n)
        first = np.cumsum(counts) - counts
        pos = np.arange(len(data)) - first[pid_codes]  # 0-based slot per person

        # per-person cumulative time -> start/end in minutes
        cum = np.cumsum(durs)
        base = np.concatenate([[0.0], cum])[
            first
        ]  # cum time before each person
        ends = cum - base[pid_codes]
        starts = ends - durs

        s_bin = np.clip(np.rint(starts / self.bin_size), 0, self.n_bins)
        e_bin = np.clip(np.rint(ends / self.bin_size), 0, self.n_bins)

        encoded = np.zeros((n, max_length, 3), dtype=np.float32)
        encoded[:, :, 0] = self.eos  # EOS marker and padding

        keep = pos < max_length - 1  # always leave room for one EOS
        rows, cols = pid_codes[keep], pos[keep]
        encoded[rows, cols, 0] = acts[keep]
        encoded[rows, cols, 1] = s_bin[keep]
        encoded[rows, cols, 2] = e_bin[keep]

        return torch.from_numpy(encoded)

    # ------------------------------------------------------------------ decode
    def decode(
        self, schedules: Tensor, argmax=True, chunk_size: int | None = None
    ) -> pd.DataFrame:
        """Decode into a DataFrame of pid | act | start | end | duration.

        Args:
            schedules: if argmax, per-class scores/one-hots [N, L, E + 2K] where the
                last dim is [act classes | start classes | end classes]; otherwise
                index tensor [N, L, 3] of (act, start_bin, end_bin).

        Schedules are rebuilt to be contiguous from the predicted *end* bins
        (start of event j = end of event j-1, ends forced non-decreasing); the start
        head is auxiliary. The usual fix_durations option is then applied.
        """
        K = self.time_classes
        with torch.no_grad():
            if argmax:
                acts, _, ends = torch.split(
                    schedules, [self.encodings, K, K], dim=-1
                )
                acts, ends = acts.argmax(dim=-1), ends.argmax(dim=-1)
            else:
                acts, _, ends = schedules.unbind(dim=-1)
            acts = acts.detach().to("cpu").numpy()
            ends = ends.detach().to("cpu").numpy()
        del schedules

        names = np.empty(self.encodings, dtype=object)
        for i, a in self.index_to_acts.items():
            names[i] = a

        N = acts.shape[0]
        step = chunk_size or N
        frames = [
            self._decode_chunk(acts[s : s + step], ends[s : s + step], s, names)
            for s in range(0, N, step)
        ]
        df = pd.concat(frames, ignore_index=True)
        return self._fix(df)

    def _decode_chunk(
        self,
        acts: np.ndarray,
        end_bins: np.ndarray,
        offset: int,
        names: np.ndarray,
    ) -> pd.DataFrame:
        acts = acts.astype(np.int64)
        end_bins = end_bins.astype(np.int64)

        is_eos = acts == self.eos
        eos_seen = np.cumsum(is_eos, axis=1)
        valid = (eos_seen == 0) & (acts != self.sos)

        end_bins = np.where(valid, end_bins, 0)
        end_bins = np.minimum(
            np.maximum.accumulate(end_bins, axis=1), self.n_bins
        )
        prev_bins = np.concatenate(
            [
                np.zeros((end_bins.shape[0], 1), dtype=np.int64),
                end_bins[:, :-1],
            ],
            axis=1,
        )
        ends = np.rint(end_bins * self.bin_size).astype(np.int64)
        starts = np.rint(prev_bins * self.bin_size).astype(np.int64)
        dur = np.where(valid, ends - starts, 0)

        # empty plan hack (as in ContinuousEncoder): EOS reached with no time spent
        empty = is_eos.any(axis=1) & (dur.sum(axis=1) == 0)
        if empty.any():
            print(
                f"Failed to decode {int(empty.sum())} plans in chunk, returning empty plan"
            )
        home = is_eos & (eos_seen == 1) & empty[:, None]

        keep = valid & ~empty[:, None]
        rows, cols = np.nonzero(keep | home)
        act_names = names[acts[rows, cols]]
        act_names[home[rows, cols]] = "home"

        return pd.DataFrame(
            {
                "pid": rows + offset,
                "act": act_names,
                "start": np.where(home[rows, cols], 0, starts[rows, cols]),
                "end": np.where(home[rows, cols], 0, ends[rows, cols]),
            }
        )

    def _fix(self, df: pd.DataFrame) -> pd.DataFrame:
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
