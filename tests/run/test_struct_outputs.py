"""A step returning a struct or a list of records, declared `Struct[Item]` / `Columnar[Item]`."""
from __future__ import annotations

from typing import TypedDict

import polars as pl
import pytest

from decider import Columnar, Engine, Struct, flow, loop, step
from decider.testing import assert_equivalent


class Product(TypedDict):
    rate: float
    code: int


@step(output="offer")
def best(rate: float, code: int) -> Struct[Product]:
    return {"rate": rate, "code": code}


@step(output="offer")
def best_tuple(rate: float, code: int) -> Struct[Product]:
    return rate, code


@step(output="offers")
def ladder(code: int) -> Columnar[Product]:
    return [{"rate": 0.1, "code": code}, {"rate": 0.2, "code": code + 1}]


@step(output="offers")
def ladder_tuple(code: int) -> Columnar[Product]:
    return [(0.1, code), (0.2, code + 1)]


def reads_offer(offer: Struct[Product]) -> float:
    return offer["rate"] * 10.0 + offer["code"]


def reads_offers(offers: Columnar[Product]) -> float:
    total = 0.0
    for j in range(len(offers.rate)):
        total += offers.rate[j]
    return total


class Tagged(TypedDict):
    label: str


@step(output="items")
def tagged(n: int) -> Columnar[Tagged]:
    return [{"label": "a"}] * n


def test_a_struct_output_keeps_its_shape_on_empty_rows():
    exe = Engine().bind(flow(best, name="order"), mode="fused")
    out = exe.run(pl.DataFrame({"rate": [0.1, 0.2], "code": [1, 2]}))
    assert out["offer"].dtype == pl.Struct({"rate": pl.Float64, "code": pl.Int64})
    assert out["offer"].to_list() == [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}]
    empty = exe.run(pl.DataFrame({"rate": [], "code": []}))
    assert empty["offer"].dtype == pl.Struct({"rate": pl.Float64, "code": pl.Int64})


def test_a_tuple_returning_struct_fuses_into_the_shared_kernel():
    exe = Engine().bind(flow(best_tuple, name="order"), mode="fused")
    out = exe.run(pl.DataFrame({"rate": [0.1, 0.2], "code": [1, 2]}))
    assert out["offer"].to_list() == [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}]
    assert exe.fallbacks() == {}


def test_a_struct_a_later_step_reads_splits_the_kernel_and_agrees_across_modes():
    out = assert_equivalent(flow(best_tuple, reads_offer, name="order").emit("offer"),
                            pl.DataFrame({"rate": [0.1, 0.2], "code": [1, 2]}))
    assert out["reads_offer"].to_list() == [2.0, 4.0]
    assert out["offer"].to_list() == [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}]


def test_a_struct_can_be_a_loop_carry_and_packs_into_one_kernel():
    @step(output="going")
    def more(rate: float) -> bool:
        return rate < 3.0

    @step(outputs=("offer", "rate"))
    def make(rate: float, code: int) -> tuple[Struct[Product], float]:
        return (rate + 1.0, code), rate + 1.0

    @step(output="rate")
    def advance(rate: float) -> float:
        return rate + 1

    search = loop(more, flow(make, advance, name="body"), carries=["offer", "rate"],
                  max_iterations=5, name="search")
    exe = Engine().bind(flow(search, name="p"), mode="fused")
    out = exe.run(pl.DataFrame({"rate": [0.0], "code": [7]}))
    assert out["offer"].to_list() == [{"rate": 3.0, "code": 7}]
    assert "p/search" in exe.runner.packed


def test_a_dict_returning_struct_stays_on_the_python_path_with_a_reason():
    exe = Engine().bind(flow(best, name="order"), mode="fused")
    assert exe.run(pl.DataFrame({"rate": [0.1], "code": [1]}))["offer"].to_list() == [
        {"rate": 0.1, "code": 1},
    ]
    assert "return a tuple of the fields" in exe.fallbacks()["order/best"]


def test_a_columnar_output_keeps_its_shape_on_empty_rows():
    exe = Engine().bind(flow(ladder, name="order"), mode="fused")
    out = exe.run(pl.DataFrame({"code": [1]}))
    assert out["offers"].dtype == pl.List(pl.Struct({"rate": pl.Float64, "code": pl.Int64}))
    assert out["offers"].to_list() == [[{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}]]
    empty = exe.run(pl.DataFrame({"code": []}))
    assert empty["offers"].dtype == pl.List(pl.Struct({"rate": pl.Float64, "code": pl.Int64}))


def test_a_dict_returning_columnar_output_stays_on_the_python_path_with_a_reason():
    exe = Engine().bind(flow(ladder, name="order"), mode="fused")
    assert exe.run(pl.DataFrame({"code": [1]}))["offers"].to_list() == [
        [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}],
    ]
    assert "return a list of tuples" in exe.fallbacks()["order/ladder"]


def test_a_tuple_returning_columnar_output_fuses_into_the_shared_kernel():
    exe = Engine().bind(flow(ladder_tuple, name="order"), mode="fused")
    out = exe.run(pl.DataFrame({"code": [1, 5]}))
    assert out["offers"].to_list() == [
        [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}],
        [{"rate": 0.1, "code": 5}, {"rate": 0.2, "code": 6}],
    ]
    assert exe.fallbacks() == {}


@pytest.mark.parametrize("mode", ("fused", "stepped"))
def test_a_columnar_output_a_later_step_reads_it_and_the_kernel_splits(mode):
    # Not `assert_equivalent`'s full mode list: a step-produced (not frame-sourced) `Columnar[...]`
    # value has no per-row representation in `mode="interpreted"`, a pre-existing limitation this
    # feature doesn't touch.
    exe = Engine().bind(flow(ladder_tuple, reads_offers, name="order").emit("offers"), mode=mode)
    out = exe.run(pl.DataFrame({"code": [1, 5]}))
    assert out["reads_offers"].to_list() == [pytest.approx(0.3), pytest.approx(0.3)]
    assert out["offers"].to_list() == [
        [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}],
        [{"rate": 0.1, "code": 5}, {"rate": 0.2, "code": 6}],
    ]
    assert exe.fallbacks() == {}


def test_a_columnar_output_grows_correctly_past_its_initial_capacity():
    @step(output="items")
    def variable(n: int) -> Columnar[Product]:
        return [(float(i), i) for i in range(n % 5)]

    exe = Engine().bind(flow(variable, name="p"), mode="fused")
    n = list(range(500))
    out = exe.run(pl.DataFrame({"n": n}))
    assert exe.fallbacks() == {}
    assert [len(row) for row in out["items"].to_list()] == [k % 5 for k in n]


def test_a_nullable_columnar_output_stays_on_the_python_path_with_a_reason():
    @step(output="items")
    def maybe(n: int) -> Columnar[Product] | None:
        return None if n == 0 else [(1.0, n)]

    exe = Engine().bind(flow(maybe, name="p"), mode="fused")
    out = exe.run(pl.DataFrame({"n": [0, 1]}))
    assert out["items"].to_list() == [None, [{"rate": 1.0, "code": 1}]]
    assert "return a plain Columnar[...]" in exe.fallbacks()["p/maybe"]


def test_a_columnar_output_item_field_of_an_unsupported_type_stays_on_the_python_path():
    exe = Engine().bind(flow(tagged, name="p"), mode="fused")
    out = exe.run(pl.DataFrame({"n": [2]}))
    assert out["items"].to_list() == [[{"label": "a"}, {"label": "a"}]]
    assert "must be float, int or bool" in exe.fallbacks()["p/tagged"]
