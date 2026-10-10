"""A one-arm branch: false rows run nothing and keep their earlier `modifies` values."""
import polars as pl
import pytest

from decikit import branch, flow, param, step
from decikit._engine import Engine
from decikit.exceptions import WiringError
from decikit.lab.testing import assert_equivalent


def eligible(score: float) -> bool:
    return score > 50.0


@step(output="limit")
def full_limit(limit: float, bonus: float = param(1000.0)) -> float:
    return limit + bonus


@step(output="limit")
def set_limit(score: float) -> float:
    return 1.0 + 0.0 * score


LIMITS = pl.DataFrame({"score": [60.0, 10.0], "limit": [3000.0, 0.0]})


def test_a_true_row_runs_the_single_arm():
    rules = branch(eligible, full_limit, modifies=["limit"], name="rules")
    out = assert_equivalent(flow(rules), LIMITS)
    assert out["limit"].to_list() == [4000.0, 0.0]


def test_a_false_row_keeps_its_earlier_modified_value():
    rules = branch(eligible, full_limit, modifies=["limit"], name="rules")
    out = assert_equivalent(flow(rules), LIMITS)
    assert out["limit"].to_list()[1] == LIMITS["limit"].to_list()[1]


def test_a_false_row_keeps_a_value_written_before_the_branch():
    rules = branch(eligible, full_limit, modifies=["limit"], name="rules")
    out = assert_equivalent(flow(set_limit, rules), LIMITS.drop("limit"))
    assert out["limit"].to_list() == [1001.0, 1.0]


def test_a_single_arm_with_an_int_condition_is_an_error():
    def band(b: int) -> int:
        return b

    with pytest.raises(WiringError, match="a single arm needs a bool condition"):
        flow(branch(band, full_limit, modifies=["limit"], name="rules")).run(LIMITS.with_columns(b=pl.lit(0)))


def test_zero_arms_is_an_error():
    with pytest.raises(WiringError, match="at least one arm"):
        branch(eligible, modifies=["limit"], name="rules")
