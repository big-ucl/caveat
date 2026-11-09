from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from caveat.encoding.base import BaseDataset, BaseEncoder
from caveat.encoding.cont_weighting import (
    act_weight_library,
    seq_weight_library,
)
from caveat.encoding.continuous import fix_end_durations, norm_durations


class EventEncoder(BaseEncoder):
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

    dataset = BaseDataset

    def __init__(
        self,
        max_length: int = 12,
        norm_duration: int = 1440,
        n_bins: int = 96,
        **kwargs,
    ):
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

        # decoding: "accumulate" (contiguous from end bins, running max) or
        # "overlay" (paint each event at its own [start, end), later events
        # overwrite earlier ones, no running max)
        self.decode_mode = kwargs.get("decode_mode", "accumulate")
        if self.decode_mode not in ("accumulate", "overlay"):
            raise ValueError(
                f"Unknown decode_mode: {self.decode_mode}, should be 'accumulate' or 'overlay'"
            )
        # overlay only: fill uncovered bins by extending the previous event
        # (and the first event backwards to 0). False leaves gaps.
        self.gap_fill = kwargs.get("gap_fill", True)

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
        max_length: {self.max_length}
        fix_durations: {self.fix_durations}
        decode_mode: {self.decode_mode}
        gap_fill: {self.gap_fill}
        event_weighting: {self.event_weighting} (power {self.weight_power}, min count {self.min_count}, max weight {self.max_weight})
        weighting: {self.weighting}, joint_weighting: {self.joint_weighting}
        trim_eos: {self.trim_eos}
        jitter: {self.jitter}
        """
        )

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

        decode_mode="accumulate": schedules are rebuilt to be contiguous from the
        predicted *end* bins (start of event j = end of event j-1, ends forced
        non-decreasing); the start head is not used.

        decode_mode="overlay": every event is placed at its own predicted
        [start, end) and later events overwrite (crop) earlier ones. Events
        with end <= start vanish. No running maximum, so one late end bin can
        no longer push the following events later.

        The usual fix_durations option is then applied.
        """
        K = self.time_classes
        with torch.no_grad():
            if argmax:
                acts, starts, ends = torch.split(
                    schedules, [self.encodings, K, K], dim=-1
                )
                acts, starts, ends = (
                    acts.argmax(dim=-1),
                    starts.argmax(dim=-1),
                    ends.argmax(dim=-1),
                )
            else:
                acts, starts, ends = schedules.unbind(dim=-1)
            acts = acts.detach().to("cpu").numpy()
            starts = starts.detach().to("cpu").numpy()
            ends = ends.detach().to("cpu").numpy()
        del schedules

        names = np.empty(self.encodings, dtype=object)
        for i, a in self.index_to_acts.items():
            names[i] = a

        N = acts.shape[0]
        step = chunk_size or N
        if self.decode_mode == "overlay":
            frames = [
                self._decode_chunk_overlay(
                    acts[s : s + step],
                    starts[s : s + step],
                    ends[s : s + step],
                    s,
                    names,
                )
                for s in range(0, N, step)
            ]
        else:
            frames = [
                self._decode_chunk(
                    acts[s : s + step], ends[s : s + step], s, names
                )
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

    def _decode_chunk_overlay(
        self,
        acts: np.ndarray,
        start_bins: np.ndarray,
        end_bins: np.ndarray,
        offset: int,
        names: np.ndarray,
    ) -> pd.DataFrame:
        """Paint events onto a [n, n_bins] timeline in order (later wins), then
        run-length encode. Event *indices* are painted, not activity types, so
        consecutive events of the same type stay separate."""
        B = self.n_bins
        acts = acts.astype(np.int64)
        s = np.clip(start_bins.astype(np.int64), 0, B)
        e = np.clip(end_bins.astype(np.int64), 0, B)
        n, L = acts.shape

        is_eos = acts == self.eos
        valid = (np.cumsum(is_eos, axis=1) == 0) & (acts != self.sos)

        bins = np.arange(B)[None, :]
        timeline = np.full((n, B), -1, dtype=np.int64)
        for j in range(L):  # event j covers bins s_j .. e_j - 1
            paint = (
                valid[:, j : j + 1]
                & (bins >= s[:, j : j + 1])
                & (bins < e[:, j : j + 1])
            )
            timeline = np.where(paint, j, timeline)

        if self.gap_fill:
            painted = timeline >= 0
            fwd = np.maximum.accumulate(np.where(painted, bins, -1), axis=1)
            bwd = np.minimum.accumulate(
                np.where(painted, bins, B)[:, ::-1], axis=1
            )[:, ::-1]
            src = np.minimum(np.where(fwd >= 0, fwd, bwd), B - 1)
            timeline = np.take_along_axis(timeline, src, axis=1)

        # run-length encode: a segment starts where the event index changes
        prev = np.concatenate(
            [np.full((n, 1), -2, dtype=np.int64), timeline[:, :-1]], axis=1
        )
        nxt = np.concatenate(
            [timeline[:, 1:], np.full((n, 1), -2, dtype=np.int64)], axis=1
        )
        seg_start = (timeline != prev) & (timeline >= 0)
        seg_end = (timeline != nxt) & (timeline >= 0)
        rows, first_bin = np.nonzero(seg_start)
        rows_e, last_bin = np.nonzero(seg_end)
        assert len(rows) == len(rows_e)

        ev = timeline[rows, first_bin]
        out = pd.DataFrame(
            {
                "pid": rows + offset,
                "act": names[acts[rows, ev]],
                "start": np.rint(first_bin * self.bin_size).astype(np.int64),
                "end": np.rint((last_bin + 1) * self.bin_size).astype(np.int64),
            }
        )

        # empty plan hack (as in accumulate mode): nothing painted
        empty = np.nonzero(~(timeline >= 0).any(axis=1))[0]
        if len(empty):
            print(
                f"Failed to decode {len(empty)} plans in chunk, returning empty plan"
            )
            out = pd.concat(
                [
                    out,
                    pd.DataFrame(
                        {
                            "pid": empty + offset,
                            "act": "home",
                            "start": 0,
                            "end": 0,
                        }
                    ),
                ],
                ignore_index=True,
            ).sort_values(["pid", "start"], kind="stable", ignore_index=True)
        return out

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
