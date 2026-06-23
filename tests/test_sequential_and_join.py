"""Tests for SequentialModule (|) and JoinModule."""

import polars as pl
import pytest

from decider.modules.functional import generate_from_functions
from decider.modules.primitives.join import JoinModule
from decider.modules.primitives.sequential import SequentialModule


# ── fixtures ──────────────────────────────────────────────────────────────────

def _scorer():
    def score(amount: pl.Expr) -> pl.Expr:
        return amount * 2

    M = generate_from_functions("scorer", score)
    return M(name="scorer")


def _flagger():
    def flag(score: pl.Expr) -> pl.Expr:
        return score > 10

    M = generate_from_functions("flagger", flag)
    return M(name="flagger")


@pytest.fixture
def single_df():
    return pl.DataFrame({"amount": [3.0, 6.0, 9.0]})


@pytest.fixture
def txns_users():
    txns = pl.DataFrame({
        "txn_id": [1, 2, 3],
        "user_id": [1, 2, 1],
        "amount": [10.0, 20.0, 30.0],
    })
    users = pl.DataFrame({
        "user_id": [1, 2],
        "tier": ["gold", "silver"],
    })
    return txns, users


# ── sequential ────────────────────────────────────────────────────────────────

def test_pipe_creates_sequential(single_df):
    s = _scorer()
    f = _flagger()
    pipeline = s | f
    assert isinstance(pipeline, SequentialModule)
    assert len(pipeline.steps) == 2


def test_sequential_output_is_last_step(single_df):
    s = _scorer()
    f = _flagger()
    result = (s | f)({"input": single_df})
    assert "flag" in result.columns
    # score should be carried through since it's an input column
    assert "score" in result.columns


def test_sequential_chaining(single_df):
    def a(amount: pl.Expr) -> pl.Expr:
        return amount + 1

    def b(a: pl.Expr) -> pl.Expr:
        return a * 10

    def c(b: pl.Expr) -> pl.Expr:
        return b - 5

    A = generate_from_functions("a_mod", a)(name="a")
    B = generate_from_functions("b_mod", b)(name="b")
    C = generate_from_functions("c_mod", c)(name="c")

    result = (A | B | C)({"input": single_df})
    # amount=3 → a=4 → b=40 → c=35
    assert result["c"][0] == pytest.approx(35.0)


def test_sequential_input_frame_keys_from_first_step():
    s = _scorer()
    f = _flagger()
    pipeline = s | f
    assert pipeline.get_input_frame_keys() == ["input"]


def test_pipe_extends_existing_sequential(single_df):
    s = _scorer()
    f = _flagger()
    seq = s | f
    # piping again should extend steps, not nest
    extra = seq | s
    assert isinstance(extra, SequentialModule)
    assert len(extra.steps) == 3


# ── join ──────────────────────────────────────────────────────────────────────

def test_join_basic(txns_users):
    txns, users = txns_users
    join = JoinModule(name="j", left="txns", right="users", on="user_id", how="left")
    result = join({"txns": txns, "users": users})
    assert "tier" in result.columns
    assert len(result) == 3


def test_join_inner_drops_unmatched():
    left = pl.DataFrame({"id": [1, 2, 3], "val": [10.0, 20.0, 30.0]})
    right = pl.DataFrame({"id": [1, 2], "label": ["a", "b"]})
    join = JoinModule(name="j", left="left", right="right", on="id", how="inner")
    result = join({"left": left, "right": right})
    assert len(result) == 2


def test_join_input_frame_keys(txns_users):
    txns, users = txns_users
    join = JoinModule(name="j", left="txns", right="users", on="user_id", how="left")
    keys = join.get_input_frame_keys()
    assert "txns" in keys
    assert "users" in keys


def test_join_then_expression(txns_users):
    txns, users = txns_users
    join = JoinModule(name="j", left="txns", right="users", on="user_id", how="left")

    def doubled(amount: pl.Expr) -> pl.Expr:
        return amount * 2

    M = generate_from_functions("doubled", doubled)
    pipeline = join | M(name="scorer")
    result = pipeline({"txns": txns, "users": users})
    assert "doubled" in result.columns
    assert "tier" in result.columns
