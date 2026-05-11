import numpy as np


class ActivityGen:
    duration = 24 * 60  # minutes
    step_size = 15  # minutes
    possible_states = ["home", "work", "shop", "education", "leisure"]
    initial_state = 0  # assume fixed start of home
    pivot_adjustment = 60  # config is in hours

    transition_config = (
        {  # (hour, ~weights) these are interpolated for each time step
            "home": {
                "home": [(0, 100), (5, 100), (11, 0.1), (23, 100), (24, 100)],
                "work": [(0, 0), (6, 0), (9, 0.2), (11, 0.1), (17, 0), (24, 0)],
                "shop": [(0, 0), (6, 0), (7, 2), (11, 1), (20, 0), (24, 0)],
                "education": [
                    (0, 0),
                    (7.5, 0),
                    (8.5, 5),
                    (11, 0.01),
                    (17, 0.01),
                    (20, 0),
                    (24, 0),
                ],
                "leisure": [
                    (0, 0),
                    (6, 0),
                    (9, 2),
                    (16, 0.1),
                    (22, 0),
                    (24, 0),
                ],
            },
            "work": {
                "home": [
                    (0, 0),
                    (12, 0),
                    (13, 0.2),
                    (16, 0.5),
                    (17, 1),
                    (24, 100),
                ],
                "work": [(0, 100), (12, 100), (20, 0), (24, 0)],
                "shop": [
                    (0, 0),
                    (12, 0),
                    (13, 0.1),
                    (14, 0),
                    (18, 0.1),
                    (19, 0),
                    (24, 0),
                ],
                "education": [
                    (0, 0),
                    (12, 0),
                    (13, 0.1),
                    (14, 0),
                    (16, 0),
                    (17, 0.1),
                    (19, 0),
                    (24, 0),
                ],
                "leisure": [(0, 0), (15, 0), (16, 0.1), (17, 0.2), (24, 0)],
            },
            "shop": {
                "home": [(0, 0.3), (23, 1), (24, 1)],
                "work": [(0, 0.1), (14, 0.1), (15, 0), (24, 0)],
                "shop": [(0, 10), (15, 10), (16, 0), (24, 0)],
                "education": [(0, 0.1), (15, 0), (24, 0)],
                "leisure": [(0, 0.2), (15, 0), (24, 0)],
            },
            "education": {
                "home": [
                    (0, 0),
                    (12, 0),
                    (13, 0.2),
                    (16, 0.1),
                    (17, 100),
                    (24, 100),
                ],
                "work": [(0, 0), (12, 1), (15, 0), (24, 0)],
                "shop": [(0, 0), (6, 0), (7, 0.1), (11, 0.3), (23, 0), (24, 0)],
                "education": [(0, 100), (12, 100), (17, 100), (18, 0), (24, 0)],
                "leisure": [
                    (0, 0),
                    (6, 0),
                    (9, 0.1),
                    (16, 0.1),
                    (17, 0),
                    (24, 0),
                ],
            },
            "leisure": {
                "home": [
                    (0, 0),
                    (12, 0),
                    (13, 0.2),
                    (16, 0.1),
                    (17, 100),
                    (24, 100),
                ],
                "work": [(0, 1), (12, 1), (23, 0), (24, 0)],
                "shop": [(0, 0), (6, 0), (7, 0.1), (11, 0.3), (23, 0), (24, 0)],
                "education": [(0, 0), (24, 0)],
                "leisure": [(0, 100), (19, 100), (23, 0), (24, 0)],
            },
        }
    )

    repetition_tollerance = np.array(
        [10, 1, 1, 1, 2]
    )  # tollerance for repeats before transition penalty
    repetition_sensitivity = np.array([1, 2, 1, 2, 1])  # size of penalty

    min_duration_tollerance = np.array(
        [180, 420, 60, 120, 60]
    )  # tollerance for min duration before penalty
    min_duration_sensitivity = np.array(
        [1.0, 1.2, 1.0, 1.2, 1.0]
    )  # size of penalty

    max_duration_tollerance = np.array(
        [12 * 60, 6 * 60, 60, 360, 120]
    )  # tollerance for duration before transition penalty, minutes
    max_duration_sensitivity = np.array(
        [0.1, 0.1, 0.1, 0.1, 0.1]
    )  # size of penalty

    def __init__(self):
        self.map = {i: s for i, s in enumerate(self.possible_states)}
        self.steps = self.duration // self.step_size
        self.transition_weights = None

    def build(self, config=None):
        if config is None:
            config = self.transition_config
        num_states = len(self.possible_states)
        self.transition_weights = np.zeros((num_states, num_states, self.steps))
        for i in range(num_states):
            in_state = self.possible_states[i]
            state_transitions = config[in_state]
            for j in range(num_states):
                out_state = self.possible_states[j]
                pivots = state_transitions[out_state]
                self.transition_weights[i][j] = interpolate_from_pivots(
                    pivots, self.steps, self.pivot_adjustment, self.step_size
                )

        self.transition_weights = np.transpose(
            self.transition_weights, (0, 2, 1)
        )  # ie [in_state, minute, out_state]

    def run(self):
        """_summary_"""
        trace = []  # [(act, start, end, dur), (act, start, end, dur), ...]
        state = self.initial_state
        activity_counts = np.zeros((len(self.possible_states)))
        activity_counts[state] += 1
        activity_durations = np.zeros((len(self.possible_states)))
        activity_durations[state] += self.step_size

        for step in range(1, self.steps):
            p = self.transition_probabilities(
                state, step, activity_counts, activity_durations
            )
            new_state = np.random.choice(len(self.possible_states), p=p)
            if new_state != state:
                time = step * self.step_size
                if not trace:  # first transition
                    prev_end = 0
                else:
                    prev_end = trace[-1][2]
                trace.append((state, prev_end, time, time - prev_end))

                # update state
                state = new_state
                activity_counts[state] += 1
                activity_durations[state] = 0  # reset

            activity_durations[state] += self.step_size

        # close
        prev_end = trace[-1][2]
        trace.append((state, prev_end, self.duration, self.duration - prev_end))
        return trace

    def transition_probabilities(
        self,
        state,
        step,
        activity_counts: np.ndarray,
        activity_durations: np.ndarray,
    ):
        p = self.transition_weights[state][step]
        p = (
            p
            * self.repeat_adjustment(activity_counts)
            * self.min_duration_adjustment(activity_durations)
            * self.max_duration_adjustment(activity_durations)
        )
        return p / sum(p)

    def repeat_adjustment(self, activity_counts: np.ndarray) -> np.ndarray:
        """Penalise activities based on how often they have been done.

        Args:
            activity_counts (np.ndarray): counts of activity repetitions

        Returns:
            np.ndarray: transition factor adjustments
        """
        return 1 / (
            np.clip((activity_counts - self.repetition_tollerance), 1, None)
            ** self.repetition_sensitivity
        )

    def max_duration_adjustment(
        self, activity_durations: np.ndarray
    ) -> np.ndarray:
        """Penalise current activity based on duration.

        Args:
            activity_durations (np.ndarray): activity durations

        Returns:
            np.ndarray: transition factor adjustments
        """
        return 1 / (
            np.clip(
                (activity_durations - self.max_duration_tollerance), 1, None
            )
            ** self.max_duration_sensitivity
        )

    def min_duration_adjustment(
        self, activity_durations: np.ndarray
    ) -> np.ndarray:
        """Penalise current activity based on duration.

        Args:
            activity_durations (np.ndarray): activity durations

        Returns:
            np.ndarray: transition factor adjustments
        """
        return (
            np.clip(
                (self.min_duration_tollerance - activity_durations), 1, None
            )
            ** self.min_duration_sensitivity
        )


def interpolate_from_pivots(
    pivots: list[tuple[float, float]],
    size: int = 1440,
    pivot_adjustment: int = 60,
    step_size: int = 1,
) -> np.ndarray:
    """Create a descretised array of shape 'size' based on given 'pivots'.

    Args:
        pivots (list[tuple[float, float]]): _description_
        size (int, optional): _description_. Defaults to 1440
        pivot_adjustment (int, optional): _description_. Defaults to 60
        step_size (int, optional): Defaults to 1

    Returns:
        np.ndarray: bins
    """
    bins = np.zeros((size), dtype=np.float64)
    for k in range(len(pivots) - 1):
        a_pivot, a_value = pivots[k]
        b_pivot, b_value = pivots[k + 1]
        a_pivot = int(a_pivot * pivot_adjustment / step_size)
        b_pivot = int(b_pivot * pivot_adjustment / step_size)
        a = (a_pivot, a_value)
        b = (b_pivot, b_value)
        bins[slice(a_pivot, b_pivot)] = interpolate_pivot(a, b)
    return bins


def interpolate_pivot(a: tuple[int, float], b: tuple[int, float]) -> np.ndarray:
    a_pivot, a_value = a
    b_pivot, b_value = b
    return np.linspace(a_value, b_value, abs(b_pivot - a_pivot), endpoint=False)
