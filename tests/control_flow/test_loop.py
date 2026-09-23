"""Loops in every mode: stopping, max_iterations, several carries, multi-step bodies, nesting."""
import polars as pl
import pytest

from decider import branch, flow, loop, param, step
from decider.engine import Engine
from decider.exceptions import WiringError
from decider.testing import assert_equivalent


def packed(pipeline, frame):
    exe = Engine().bind(pipeline, "fused")
    exe.run(frame)
    return sorted(exe.runner.packed)


def below_target(best_offer: float, target: float = param(90.0)) -> bool:
    return best_offer < target


@step(output="best_offer")
def raise_offer(best_offer: float, increment: float = param(5.0)) -> float:
    return best_offer + increment


def offers(max_iterations=511):
    return loop(below_target, raise_offer, carries=["best_offer"], max_iterations=max_iterations, name="offers")


def test_a_loop_runs_each_row_until_its_condition_fails():
    frame = pl.DataFrame({"best_offer": [0.0, 85.0, 88.0, 95.0]})
    # 0 -> 90 in 18 steps; 85 -> 90 in 1; 88 overshoots to 93; 95 never iterates.
    assert assert_equivalent(offers(), frame)["best_offer"].to_list() == [90.0, 90.0, 93.0, 95.0]


def test_a_loop_stops_at_max_iterations_without_an_error():
    frame = pl.DataFrame({"best_offer": [0.0, 85.0]})
    assert assert_equivalent(offers(max_iterations=3), frame)["best_offer"].to_list() == [15.0, 90.0]


def test_params_reach_the_condition_and_the_body():
    frame = pl.DataFrame({"best_offer": [0.0]})
    params = {"offers": {"below_target": {"target": 20.0}, "raise_offer": {"increment": 7.0}}}
    assert assert_equivalent(offers(), frame, params)["best_offer"].to_list() == [21.0]


def test_a_body_need_not_read_what_it_carries():
    @step(output="best_offer")
    def fixed(seed: float) -> float:
        return seed * 100.0

    search = loop(below_target, fixed, carries=["best_offer"], max_iterations=5, name="search")
    frame = pl.DataFrame({"best_offer": [0.0, 95.0], "seed": [1.0, 1.0]})
    assert assert_equivalent(search, frame)["best_offer"].to_list() == [100.0, 95.0]


def test_a_carry_the_body_never_writes_is_an_error():
    @step(output="other")
    def other(best_offer: float) -> float:
        return best_offer

    with pytest.raises(WiringError, match="carries 'best_offer', but the body never writes it"):
        loop(below_target, other, carries=["best_offer"], max_iterations=5, name="search").run(pl.DataFrame())


def test_max_iterations_is_required_and_positive():
    with pytest.raises(TypeError):
        loop(below_target, raise_offer, carries=["best_offer"], name="search")
    with pytest.raises(WiringError, match="positive int"):
        loop(below_target, raise_offer, carries=["best_offer"], max_iterations=0, name="search")


# --- several carries, multi-step bodies -------------------------------------------------


def below_limit(b: int, limit: int = param(100)) -> bool:
    return b < limit


@step(output="total")
def add(a: int, b: int) -> int:
    return a + b


@step(output="a")
def shift(b: int) -> int:
    return b


@step(output="b")
def take(total: int) -> int:
    return total


fibonacci = loop(below_limit, flow(add, shift, take, name="next"), carries=["a", "b"], max_iterations=100, name="fib")


def test_two_carries_update_together_through_a_multi_step_body():
    out = assert_equivalent(fibonacci, pl.DataFrame({"a": [0, 5, 200], "b": [1, 8, 300]}))
    assert out["a"].to_list() == [89, 89, 200]
    assert out["b"].to_list() == [144, 144, 300]
    assert packed(fibonacci, pl.DataFrame({"a": [0], "b": [1]})) == ["fib"]


def not_one(n: int) -> bool:
    return n != 1


def is_even(n: int) -> bool:
    return n % 2 == 0


@step(output="n")
def halve(n: int) -> int:
    return n // 2


@step(output="n")
def triple(n: int) -> int:
    return 3 * n + 1


@step(output="steps")
def count(steps: int) -> int:
    return steps + 1


def collatz(max_iterations=1000):
    parity = branch(is_even, halve, triple, modifies=["n"], name="parity")
    return loop(not_one, flow(parity, count, name="step"), carries=["n", "steps"],
                max_iterations=max_iterations, name="collatz")


START = pl.DataFrame({"n": [1, 6, 27, 7], "steps": [0, 0, 0, 0]})


def test_a_branch_nested_in_a_loop_body():
    out = assert_equivalent(collatz(), START)
    assert out["steps"].to_list() == [0, 8, 111, 16]
    assert out["n"].to_list() == [1, 1, 1, 1]
    assert packed(collatz(), START) == ["collatz", "collatz/step/parity"]


def test_max_iterations_bounds_a_loop_with_a_nested_branch():
    out = assert_equivalent(collatz(max_iterations=5), START)
    assert out["steps"].to_list() == [0, 5, 5, 5]
    assert out["n"].to_list() == [1, 8, 31, 52]


# --- a loop inside a branch inside a loop ----------------------------------------------


def high_risk(risk_flag: bool) -> bool:
    return risk_flag


def below_inner_cap(rate: float, cap: float = param(9.0)) -> bool:
    return rate < cap


@step(output="rate")
def bump(rate: float, step_size: float = param(3.0)) -> float:
    return rate + step_size


@step(output="rate")
def flat_rate(rate: float, flat: float = param(1.0)) -> float:
    return flat + 0.0 * rate


def below_outer_cap(rate: float, cap: float = param(50.0)) -> bool:
    return rate < cap


def test_a_loop_in_a_branch_in_a_loop():
    inner = loop(below_inner_cap, bump, carries=["rate"], max_iterations=20, name="inner")
    risk = branch(high_risk, inner, flat_rate, modifies=["rate"], name="risk")
    outer = loop(below_outer_cap, risk, carries=["rate"], max_iterations=20, name="outer")
    pipeline = flow(high_risk, outer)
    frame = pl.DataFrame({"risk_flag": [True, False], "rate": [0.0, 0.0]})
    # Both rows stall at an inner fixed point below the outer cap, so the outer loop runs to its bound.
    assert assert_equivalent(pipeline, frame)["rate"].to_list() == [9.0, 1.0]
    assert packed(pipeline, frame) == ["outer", "outer/risk", "outer/risk/inner"]


# --- nulls --------------------------------------------------------------------------------


def test_a_null_carry_never_iterated_stays_null_in_every_mode():
    search = loop(below_target, raise_offer, carries=["best_offer"], max_iterations=5, name="search")

    def gate(best_offer: float | None = None) -> bool:
        return best_offer is not None and best_offer < 90.0

    guarded = loop(gate, raise_offer, carries=["best_offer"], max_iterations=5, name="search")
    frame = pl.DataFrame({"best_offer": [80.0, None]})
    with pytest.raises(ValueError, match="best_offer"):
        search.run(frame)
    assert assert_equivalent(guarded, frame)["best_offer"].to_list() == [90.0, None]
