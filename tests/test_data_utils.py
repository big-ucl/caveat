from unittest.mock import MagicMock

import pandas as pd
import pytest

from caveat.data.utils import (
    gen_person,
    gen_person_conditional,
    gen_persons,
    gen_persons_conditional,
    split,
    trace_to_df,
    trace_to_pam,
)

TRACE = [(2, 0, 60, 60), (3, 60, 480, 420), (2, 480, 540, 60)]

MAPPING = {2: "home", 3: "work"}


# --- split ---


def test_split_basic():
    chunks = list(split(range(10), 3))
    all_items = [x for chunk in chunks for x in chunk]
    assert sorted(all_items) == list(range(10))


def test_split_single_chunk():
    chunks = list(split(range(5), 1))
    assert len(chunks) == 1
    assert list(chunks[0]) == list(range(5))


def test_split_more_chunks_than_items():
    chunks = list(split(range(3), 5))
    all_items = [x for chunk in chunks for x in chunk]
    assert sorted(all_items) == [0, 1, 2]
    assert all(len(list(c)) <= 1 for c in chunks)


def test_split_equal_size():
    chunks = list(split(range(9), 3))
    assert all(len(list(c)) == 3 for c in chunks)


# --- trace_to_df ---


def test_trace_to_df_columns():
    df = trace_to_df(TRACE)
    assert list(df.columns) == ["act", "start", "end", "duration"]


def test_trace_to_df_row_count():
    df = trace_to_df(TRACE)
    assert len(df) == len(TRACE)


def test_trace_to_df_values():
    df = trace_to_df(TRACE)
    assert df["act"].tolist() == [2, 3, 2]
    assert df["start"].tolist() == [0, 60, 480]
    assert df["duration"].tolist() == [60, 420, 60]


def test_trace_to_df_kwargs():
    df = trace_to_df(TRACE, pid=42, label="a")
    assert "pid" in df.columns
    assert (df["pid"] == 42).all()
    assert (df["label"] == "a").all()


def test_trace_to_df_empty():
    df = trace_to_df([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# --- trace_to_pam ---


def test_trace_to_pam_returns_plan():
    pam = pytest.importorskip("pam")
    plan = trace_to_pam(TRACE, MAPPING)
    assert isinstance(plan, pam.activity.Plan)


def test_trace_to_pam_activity_count():
    pam = pytest.importorskip("pam")
    plan = trace_to_pam(TRACE, MAPPING)
    activities = [c for c in plan if isinstance(c, pam.activity.Activity)]
    assert len(activities) == len(TRACE)


# --- gen_person / gen_persons ---


def _mock_gen():
    gen = MagicMock()
    gen.run.return_value = TRACE
    return gen


def test_gen_person_returns_dataframe():
    df = gen_person(_mock_gen(), pid=1)
    assert isinstance(df, pd.DataFrame)


def test_gen_person_pid_column():
    df = gen_person(_mock_gen(), pid=7)
    assert "pid" in df.columns
    assert (df["pid"] == 7).all()


def test_gen_person_row_count():
    df = gen_person(_mock_gen(), pid=1)
    assert len(df) == len(TRACE)


def test_gen_persons_concatenates():
    df = gen_persons(_mock_gen(), pids=[0, 1, 2])
    assert len(df) == 3 * len(TRACE)
    assert df["pid"].nunique() == 3


def test_gen_persons_resets_index():
    df = gen_persons(_mock_gen(), pids=[0, 1])
    assert df.index.tolist() == list(range(len(df)))


# --- gen_person_conditional / gen_persons_conditional ---


def test_gen_person_conditional_columns():
    gen = _mock_gen()
    gens = [gen, gen, gen, gen]
    df = gen_person_conditional(gens, pid=1)
    assert isinstance(df, pd.DataFrame)
    for col in ("pid", "age", "gender", "employment"):
        assert col in df.columns


def test_gen_person_conditional_pid():
    gen = _mock_gen()
    gens = [gen, gen, gen, gen]
    df = gen_person_conditional(gens, pid=99)
    assert (df["pid"] == 99).all()


def test_gen_person_conditional_row_count():
    gen = _mock_gen()
    gens = [gen, gen, gen, gen]
    df = gen_person_conditional(gens, pid=1)
    assert len(df) == len(TRACE)


def test_gen_person_conditional_employment_values():
    import random

    random.seed(0)
    gen = _mock_gen()
    gens = [gen, gen, gen, gen]
    valid = {"FTW", "PTW", "NEET", "FTE"}
    for _ in range(2000):
        df = gen_person_conditional(gens, pid=1)
        assert df["employment"].iloc[0] in valid


def test_gen_persons_conditional_concatenates():
    gen = _mock_gen()
    gens = [gen, gen, gen, gen]
    df = gen_persons_conditional(gens, pids=[0, 1, 2])
    assert len(df) == 3 * len(TRACE)
    assert df.index.tolist() == list(range(len(df)))
