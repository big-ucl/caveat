import numpy as np
from torch import Tensor, rand


class ScheduleAugment:
    def __init__(self) -> None:
        raise NotImplementedError

    def __call__(self, schedule: Tensor) -> Tensor:
        raise NotImplementedError


class SequenceJitter(ScheduleAugment):
    def __init__(self, jitter: float = 0.1):
        """Augment a sequence by adding jitter to the duration of each activity.
        Note that jitter defines the maximum delay or advance of the activity duration
        as a proportion. But note that during normalisation of the total plan duration
        this can be exceeded.

        Args:
            jitter (float, optional): activity duration maximum delta. Defaults to 0.1.
        """
        self.jitter = jitter

    def __call__(self, sequence):
        if self.jitter == 0:
            return sequence
        # ignore sos and eos (assumes 0 and 1 tokens)
        mask = sequence[:, 0] > 1
        if mask.sum() < 2:  # single activity sequences are not jittered
            return sequence
        j = (rand((sequence.shape[0])) - 0.5) * self.jitter
        deltas = j * sequence[:, 1]
        deltas -= mask * deltas[mask].mean()
        new = sequence.clone()
        new[:, 1] += deltas
        return new


class DiscreteSingleJitter(ScheduleAugment):
    def __init__(self, step_size: int, jitter: int = 0):

        self.step_size = step_size
        self.jitter = jitter

    def __call__(self, sequence) -> np.ndarray:

        transitions = np.where(sequence[:-1] != sequence[1:])[0]

        if not len(transitions):  # no transitions to jitter
            return sequence

        sequence = sequence.clone()
        idx = np.random.choice(len(transitions))
        transition_idx = transitions[idx]
        acts = sequence[transition_idx], sequence[transition_idx + 1]
        direction = np.random.choice([0, 1])
        act = acts[direction]

        durations = [
            (j - i)
            for i, j in zip(
                np.concatenate([[0], transitions + 1]),
                np.concatenate([transitions + 1, [len(sequence)]]),
            )
        ]
        dur_a, dur_b = (durations[idx], durations[idx + 1])

        min_duration = min(dur_a, dur_b)

        delta = int(min_duration * np.random.rand() * self.jitter)

        if delta != 0:
            for j in range(1, delta + 1):
                if direction == 0:
                    sequence[transition_idx + j] = act
                else:
                    sequence[transition_idx - j + 1] = act

        return sequence


class DiscreteJitter(ScheduleAugment):
    def __init__(self, step_size: int, jitter: int = 0):

        self.step_size = step_size
        self.jitter = jitter

    def __call__(self, sequence) -> np.ndarray:

        transitions = np.where(sequence[:-1] != sequence[1:])[0]

        if not len(transitions):  # no transitions to jitter
            return sequence

        sequence = sequence.clone()
        for idx, transition_idx in enumerate(transitions):
            acts = sequence[transition_idx], sequence[transition_idx + 1]
            direction = np.random.choice([0, 1])
            act = acts[direction]

            durations = [
                (j - i)
                for i, j in zip(
                    np.concatenate([[0], transitions + 1]),
                    np.concatenate([transitions + 1, [len(sequence)]]),
                )
            ]
            dur_a, dur_b = (durations[idx], durations[idx + 1])

            min_duration = min(dur_a, dur_b)

            delta = int(min_duration * np.random.rand() * self.jitter)

            if delta != 0:
                for j in range(1, delta + 1):
                    if direction == 0:
                        sequence[transition_idx + j] = act
                    else:
                        sequence[transition_idx - j + 1] = act

        return sequence
