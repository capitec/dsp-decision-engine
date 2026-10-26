"""A step returning a struct or a list of records, declared `Struct[Item]` / `Columnar[Item]`."""
from __future__ import annotations

from typing import TypedDict

import polars as pl
import pytest

from decikit import FusedEngine, InterpretedEngine, SteppedEngine, allow_fallback, branch, flow, loop, missing_as, step
from decikit.ops import Columnar, Struct
from decikit.lab.testing import assert_equivalent


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
    exe = FusedEngine().bind(flow(best, name="order"))
    out = exe.run(pl.DataFrame({"rate": [0.1, 0.2], "code": [1, 2]}))
    assert out["offer"].dtype == pl.Struct({"rate": pl.Float64, "code": pl.Int64})
    assert out["offer"].to_list() == [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}]
    empty = exe.run(pl.DataFrame({"rate": [], "code": []}))
    assert empty["offer"].dtype == pl.Struct({"rate": pl.Float64, "code": pl.Int64})


def test_a_tuple_returning_struct_fuses_into_the_shared_kernel():
    exe = FusedEngine().bind(flow(best_tuple, name="order"))
    out = exe.run(pl.DataFrame({"rate": [0.1, 0.2], "code": [1, 2]}))
    assert out["offer"].to_list() == [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}]
    assert exe.fallbacks() == {}


def test_a_struct_a_later_step_reads_splits_the_kernel_and_agrees_across_modes():
    out = assert_equivalent(flow(best_tuple, reads_offer, name="order").emit("offer"),
                            pl.DataFrame({"rate": [0.1, 0.2], "code": [1, 2]}))
    assert out["reads_offer"].to_list() == [2.0, 4.0]
    assert out["offer"].to_list() == [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}]


def positive(rate: float) -> bool:
    return rate > 0.0


@allow_fallback
def offer_rate(offer: dict | None) -> float:
    return offer["rate"] if offer else -1.0


def test_a_struct_a_branch_leaves_null_on_some_rows_is_read_by_field_in_every_mode():
    pipeline = flow(branch(positive, best_tuple, modifies=["offer"], name="priced"), offer_rate, name="order")
    out = assert_equivalent(pipeline, pl.DataFrame({"rate": [0.1, -0.2], "code": [1, 2]}))
    assert out["offer_rate"].to_list() == [0.1, -1.0]


LABELLED = pl.DataFrame({"rate": [0.1, 0.2], "code": [1, 2], "label": ["ab", "c"]})


@allow_fallback
def offer_fields(offer: dict) -> str:
    return f"{','.join(offer)}={offer.get('rate')}"


@allow_fallback
@step(output="offer")
def best_named(rate: float, label: str) -> Struct[Product]:
    return rate, len(label)  # a `str` input keeps the step in Python


@allow_fallback
@step(output="offers")
def ladder_named(code: int, label: str) -> Columnar[Product]:
    return [(0.1, code), (0.2, len(label))]


@allow_fallback
def first_offer(offers: list) -> str:
    return f"{','.join(offers[0])}={offers[0].get('rate')}"


@pytest.mark.parametrize("make", (best_tuple, best_named), ids=("compiled", "python"))
def test_a_python_step_reads_a_struct_as_a_dict_in_every_mode(make):
    out = assert_equivalent(flow(make, offer_fields, name="order"), LABELLED)
    assert out["offer_fields"].to_list() == ["rate,code=0.1", "rate,code=0.2"]


def test_a_python_step_reads_a_python_columnar_output_as_dicts_in_every_mode():
    out = assert_equivalent(flow(ladder_named, first_offer, name="order"), LABELLED)
    assert out["first_offer"].to_list() == ["rate,code=0.1", "rate,code=0.1"]


@step(output="best")
def pass_on(offers: Columnar[Product] | None, offer: Struct[Product]) -> Struct[Product]:
    return offer  # an optional Columnar input runs one compiled call per row


@allow_fallback
def best_fields(best: dict) -> str:
    return f"{','.join(best)}={best['rate'] / 0.5}"


def test_a_per_row_compiled_step_handing_on_its_struct_input_agrees_in_every_engine():
    frame = pl.DataFrame({"offers": [[{"rate": 0.1, "code": 1}], None],
                          "offer": [{"rate": 0.5, "code": 5}, {"rate": 0.6, "code": 6}]})
    out = assert_equivalent(flow(pass_on, best_fields, name="order").emit("best"), frame)
    assert out["best"].to_list() == [{"rate": 0.5, "code": 5}, {"rate": 0.6, "code": 6}]
    assert out["best_fields"].to_list() == ["rate,code=1.0", "rate,code=1.2"]


@allow_fallback
@step(output="offer")
def best_too_long(rate: float, label: str) -> Struct[Product]:
    return rate, len(label), label


@allow_fallback
@step(output="offer")
def best_not_a_tuple(rate: float, label: str) -> Struct[Product]:
    return rate


@allow_fallback
@step(output="offers")
def ladder_too_long(code: int, label: str) -> Columnar[Product]:
    return [(0.1, code, label)]


@pytest.mark.parametrize("engine", (InterpretedEngine, SteppedEngine, FusedEngine))
@pytest.mark.parametrize("fn", (best_too_long, best_not_a_tuple, ladder_too_long))
def test_a_python_struct_step_returning_something_other_than_its_fields_is_refused_in_every_engine(fn, engine):
    with pytest.raises(ValueError, match="Input should be a valid dictionary"):
        engine().bind(flow(fn, name="order")).run(LABELLED)


MIXED = pl.DataFrame({"rate": [0.1, -0.2, 0.3], "code": [1, 2, 3], "label": ["ab", "c", "def"]})
NO_OFFER = {"rate": -9.0, "code": -9}


@allow_fallback
def filled_rate(offer: dict = missing_as(NO_OFFER)) -> float:
    return offer["rate"] * 10.0 + offer["code"]


def filled_rate_compiled(offer: Struct[Product] = missing_as(NO_OFFER)) -> float:
    return offer["rate"] * 10.0 + offer["code"]


@pytest.mark.parametrize("reader", (filled_rate, filled_rate_compiled), ids=("python_reader", "kernel_reader"))
def test_a_struct_a_branch_leaves_null_on_some_rows_is_filled_in_every_engine(reader):
    pipeline = flow(branch(positive, best_tuple, modifies=["offer"], name="priced"), reader, name="order")
    out = assert_equivalent(pipeline, MIXED)
    assert out[out.columns[-1]].to_list() == [2.0, -99.0, 6.0]


@pytest.mark.parametrize("reader", (offer_rate, reads_offer), ids=("python_reader", "kernel_reader"))
@pytest.mark.parametrize("arms, offers", [
    ((best_tuple, best_named), [{"rate": 0.1, "code": 1}, {"rate": -0.2, "code": 1}, {"rate": 0.3, "code": 3}]),
    ((best_named, best_tuple), [{"rate": 0.1, "code": 2}, {"rate": -0.2, "code": 2}, {"rate": 0.3, "code": 3}]),
], ids=("kernel_arm_first", "python_arm_first"))
def test_a_struct_one_arm_writes_in_a_kernel_and_another_in_python_agrees_in_every_engine(arms, offers, reader):
    pipeline = flow(branch(positive, *arms, modifies=["offer"], name="priced"), reader, name="order").emit("offer")
    assert assert_equivalent(pipeline, MIXED)["offer"].to_list() == offers


@step(output="going")
def below_two(rate: float) -> bool:
    return rate < 2.0


@step(outputs=("offer", "rate"))
def raise_tuple(rate: float, code: int) -> tuple[Struct[Product], float]:
    return (rate + 1.0, code), rate + 1.0


@allow_fallback
@step(outputs=("offer", "rate"))
def raise_named(rate: float, label: str) -> tuple[Struct[Product], float]:
    return (rate + 1.0, len(label) * 10), rate + 1.0


@pytest.mark.parametrize("seed, body, offers", [
    (best_named, raise_tuple, [{"rate": 2.0, "code": 7}, {"rate": 5.0, "code": 1}, {"rate": 2.5, "code": 9}]),
    (best_tuple, raise_named, [{"rate": 2.0, "code": 20}, {"rate": 5.0, "code": 8}, {"rate": 2.5, "code": 30}]),
], ids=("python_seed", "python_body"))
def test_a_struct_loop_carry_written_in_a_kernel_and_in_python_agrees_in_every_engine(seed, body, offers):
    search = loop(below_two, flow(body, name="body"), carries=["offer", "rate"], max_iterations=5, name="search")
    frame = pl.DataFrame({"rate": [0.0, 5.0, 1.5], "code": [7, 8, 9], "label": ["ab", "c", "def"]})
    out = assert_equivalent(flow(seed, search, offer_rate, name="order").emit("offer"), frame)
    assert out["offer"].to_list() == offers


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
    exe = FusedEngine().bind(flow(search, name="p"))
    out = exe.run(pl.DataFrame({"rate": [0.0], "code": [7]}))
    assert out["offer"].to_list() == [{"rate": 3.0, "code": 7}]
    assert "p/search" in exe.runner.packed


def test_a_dict_returning_struct_stays_on_the_python_path_with_a_reason():
    exe = FusedEngine().bind(flow(best, name="order"))
    assert exe.run(pl.DataFrame({"rate": [0.1], "code": [1]}))["offer"].to_list() == [
        {"rate": 0.1, "code": 1},
    ]
    assert "return a tuple of the fields" in exe.fallbacks()["order/best"]


def test_a_columnar_output_keeps_its_shape_on_empty_rows():
    exe = FusedEngine().bind(flow(ladder, name="order"))
    out = exe.run(pl.DataFrame({"code": [1]}))
    assert out["offers"].dtype == pl.List(pl.Struct({"rate": pl.Float64, "code": pl.Int64}))
    assert out["offers"].to_list() == [[{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}]]
    empty = exe.run(pl.DataFrame({"code": []}))
    assert empty["offers"].dtype == pl.List(pl.Struct({"rate": pl.Float64, "code": pl.Int64}))


def test_a_dict_returning_columnar_output_stays_on_the_python_path_with_a_reason():
    exe = FusedEngine().bind(flow(ladder, name="order"))
    assert exe.run(pl.DataFrame({"code": [1]}))["offers"].to_list() == [
        [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}],
    ]
    assert "return a list of tuples" in exe.fallbacks()["order/ladder"]


def test_a_tuple_returning_columnar_output_fuses_into_the_shared_kernel():
    exe = FusedEngine().bind(flow(ladder_tuple, name="order"))
    out = exe.run(pl.DataFrame({"code": [1, 5]}))
    assert out["offers"].to_list() == [
        [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}],
        [{"rate": 0.1, "code": 5}, {"rate": 0.2, "code": 6}],
    ]
    assert exe.fallbacks() == {}


@pytest.mark.parametrize("fn", (ladder, ladder_tuple), ids=("dicts", "tuples"))
def test_a_columnar_output_comes_back_the_same_in_every_mode(fn):
    out = assert_equivalent(flow(fn, name="order"), pl.DataFrame({"code": [1, 5]}))
    assert out["offers"].to_list() == [
        [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}],
        [{"rate": 0.1, "code": 5}, {"rate": 0.2, "code": 6}],
    ]


def test_a_columnar_output_a_later_step_reads_it_and_the_kernel_splits():
    pipeline = flow(ladder_tuple, reads_offers, name="order").emit("offers")
    out = assert_equivalent(pipeline, pl.DataFrame({"code": [1, 5]}))
    assert out["reads_offers"].to_list() == [pytest.approx(0.3), pytest.approx(0.3)]
    assert out["offers"].to_list() == [
        [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}],
        [{"rate": 0.1, "code": 5}, {"rate": 0.2, "code": 6}],
    ]
    assert FusedEngine().bind(pipeline).fallbacks() == {}
    assert SteppedEngine().bind(pipeline).fallbacks() == {}


def test_a_columnar_output_grows_correctly_past_its_initial_capacity():
    @step(output="items")
    def variable(n: int) -> Columnar[Product]:
        return [(float(i), i) for i in range(n % 5)]

    exe = FusedEngine().bind(flow(variable, name="p"))
    n = list(range(500))
    out = exe.run(pl.DataFrame({"n": n}))
    assert exe.fallbacks() == {}
    assert [len(row) for row in out["items"].to_list()] == [k % 5 for k in n]


def test_a_nullable_columnar_output_stays_on_the_python_path_with_a_reason():
    @step(output="items")
    def maybe(n: int) -> Columnar[Product] | None:
        return None if n == 0 else [(1.0, n)]

    exe = FusedEngine().bind(flow(maybe, name="p"))
    out = exe.run(pl.DataFrame({"n": [0, 1]}))
    assert out["items"].to_list() == [None, [{"rate": 1.0, "code": 1}]]
    assert "return a plain Columnar[...]" in exe.fallbacks()["p/maybe"]


def test_a_columnar_output_item_field_of_an_unsupported_type_stays_on_the_python_path():
    exe = FusedEngine().bind(flow(tagged, name="p"))
    out = exe.run(pl.DataFrame({"n": [2]}))
    assert out["items"].to_list() == [[{"label": "a"}, {"label": "a"}]]
    assert "must be float, int or bool" in exe.fallbacks()["p/tagged"]
