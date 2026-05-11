import multiprocessing as mp
import random

import pandas as pd
from pam.activity import Activity, Plan, Trip
from pam.utils import minutes_to_datetime as mtdt


def trace_to_pam(trace: list[tuple], mapping: dict):
    plan = Plan()
    for act, start, end, duration in trace:
        name = mapping[act]
        plan.add(Activity(act=name, start_time=mtdt(start), end_time=mtdt(end)))
        plan.add(Trip(mode="car", start_time=mtdt(end), end_time=mtdt(end)))
    return plan


def generate_population(gen, size: int, cores: int = None):
    if cores is None:
        cores = mp.cpu_count()

    batches = list(split(range(size), cores))

    pools = mp.Pool(cores)
    results = [
        pools.apply_async(gen_persons, args=(gen, pids)) for pids in batches
    ]
    pools.close()
    pools.join()
    results = [r.get() for r in results]
    pop = pd.concat(results, ignore_index=True)
    return pop


def gen_persons(gen, pids) -> pd.DataFrame:
    return pd.concat([gen_person(gen, pid) for pid in pids], ignore_index=True)


def gen_person(gen, pid) -> pd.DataFrame:
    trace = gen.run()
    return trace_to_df(trace, pid=pid)


def trace_to_df(trace: list[tuple], **kwargs) -> pd.DataFrame:
    df = pd.DataFrame(trace, columns=["act", "start", "end", "duration"])
    for k, v in kwargs.items():
        df[k] = v
    return df


def split(a, n):
    k, m = divmod(len(a), n)
    return (
        a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
    )


def generate_population_conditional(gens, size: int, cores: int = None):
    if cores is None:
        cores = mp.cpu_count()
    batches = list(split(range(size), cores))
    pools = mp.Pool(cores)
    results = [
        pools.apply_async(gen_persons_conditional, args=(gens, pids))
        for pids in batches
    ]
    pools.close()
    pools.join()
    results = [r.get() for r in results]
    pop = pd.concat(results, ignore_index=True)
    return pop


def gen_persons_conditional(gens, pids) -> pd.DataFrame:
    return pd.concat(
        [gen_person_conditional(gens, pid) for pid in pids], ignore_index=True
    )


def gen_person_conditional(gens, pid) -> pd.DataFrame:
    age = random.randint(5, 100)
    gender = random.choice(["M", "F"])
    employment = "NEET"
    if age < 18:
        employment = "FTE"
    elif age < 21:
        if random.random() < 0.4:
            employment = "FTE"
        elif random.random() < 0.2:
            employment = "PTW"
        elif random.random() < 0.5:
            employment = "FTW"
    elif age < 76:
        p = (100 - age) / 100
        if gender == "F":
            if random.random() < p / 2:
                employment = "FTW"
            if random.random() < p:
                employment = "PTW"
            elif random.random() < 0.1:
                employment = "FTE"
        elif gender == "M":  # pragma: no branch
            if random.random() < p:
                employment = "FTW"

    if employment == "FTW":
        gen = gens[0]
    elif employment == "PTW":
        gen = gens[1]
    elif employment == "NEET":
        gen = gens[2]
    else:
        gen = gens[3]

    trace = gen.run()
    return trace_to_df(
        trace, pid=pid, age=age, gender=gender, employment=employment
    )
