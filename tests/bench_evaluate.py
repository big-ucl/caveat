"""Benchmark harness for caveat.evaluate.

Usage:
    python tests/bench_evaluate.py --population 1000 --activities 10
    python tests/bench_evaluate.py --population 5000 --activities 15
"""

import argparse
import time

import numpy as np
from pandas import DataFrame

from caveat.data.synth import ActivityGen
from caveat.evaluate.evaluate import process_metrics


def generate_population(n_people: int, n_activities: int, seed: int = 42) -> DataFrame:
    """Generate a synthetic schedule population using ActivityGen.

    Args:
        n_people: Number of people (schedules) to generate.
        n_activities: Number of activity types to use (max 5).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns pid, act, start, end, duration.
    """
    np.random.seed(seed)
    n_activities = min(n_activities, len(ActivityGen.possible_states))

    gen = ActivityGen()
    gen.possible_states = gen.possible_states[:n_activities]
    gen.repetition_tollerance = gen.repetition_tollerance[:n_activities]
    gen.repetition_sensitivity = gen.repetition_sensitivity[:n_activities]
    gen.min_duration_tollerance = gen.min_duration_tollerance[:n_activities]
    gen.min_duration_sensitivity = gen.min_duration_sensitivity[:n_activities]
    gen.max_duration_tollerance = gen.max_duration_tollerance[:n_activities]
    gen.max_duration_sensitivity = gen.max_duration_sensitivity[:n_activities]
    # Rebuild transition config for the subset of states
    subset_config = {}
    for state in gen.possible_states:
        subset_config[state] = {
            s: gen.transition_config[state][s] for s in gen.possible_states
        }
    gen.build(config=subset_config)

    rows = []
    for pid in range(n_people):
        trace = gen.run()
        for act_idx, start, end, dur in trace:
            rows.append(
                {
                    "pid": pid,
                    "act": gen.map[act_idx],
                    "start": start,
                    "end": end,
                    "duration": dur,
                }
            )
    return DataFrame(rows)


def bench(n_people: int, n_activities: int):
    """Run the benchmark: generate data, time process_metrics."""
    print(f"Generating target population ({n_people} people, {n_activities} activities)...")
    t0 = time.perf_counter()
    target = generate_population(n_people, n_activities, seed=42)
    t_gen_target = time.perf_counter() - t0
    print(f"  target: {len(target)} rows in {t_gen_target:.3f}s")

    t0 = time.perf_counter()
    synthetic = generate_population(n_people, n_activities, seed=99)
    t_gen_synth = time.perf_counter() - t0
    print(f"  synthetic: {len(synthetic)} rows in {t_gen_synth:.3f}s")

    print(f"\nRunning process_metrics()...")
    t_start = time.perf_counter()
    process_metrics(
        synthetic_schedules={"synth": synthetic},
        target_schedules=target,
        verbose=True,
    )
    t_total = time.perf_counter() - t_start
    print(f"\nTotal process_metrics time: {t_total:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark caveat evaluate module")
    parser.add_argument(
        "--population", type=int, default=20000, help="Number of people per population"
    )
    parser.add_argument(
        "--activities", type=int, default=5, help="Number of activity types (max 5)"
    )
    args = parser.parse_args()
    bench(args.population, args.activities)
