"""`Columnar[Item]`: a list column's items as arrays per field, sliced per parent row."""
from __future__ import annotations

import gc
from datetime import date
from typing import TypedDict

import numpy as np
import polars as pl
import pytest

from decider import Engine, Columnar, branch, flow, loop, missing_as, step
from decider.engine.compile.rows import ARROW_ROWS, build_rows
from decider.engine.run.state import from_series
from decider.testing import assert_equivalent

MODES = ("interpreted", "stepped", "fused")


class Item(TypedDict):
    price: float


def bundle_total(mask: int, items: Columnar[Item]) -> float:
    total = 0.0
    for j in range(len(items.price)):
        if (mask >> j) & 1:
            total += items.price[j]
    return total


FRAME = pl.DataFrame({
    "mask": [0b11, 0b01, 0b00],
    "items": [[{"price": 2.0}, {"price": 3.0}], [{"price": 5.0}], []],
})


def test_bundle_total_agrees_across_every_mode():
    out = assert_equivalent(flow(bundle_total, name="p"), FRAME)
    assert out["bundle_total"].to_list() == [5.0, 5.0, 0.0]


@pytest.mark.parametrize("mode", MODES)
def test_a_row_with_no_items_is_zero_items_not_an_error(mode):
    exe = Engine().bind(flow(bundle_total, name="p"), mode=mode)
    assert exe.score({"mask": 0, "items": []})["bundle_total"] == 0.0


@pytest.mark.parametrize("mode", ("stepped", "fused"))
def test_rows_joins_the_shared_array_kernel(mode):
    exe = Engine().bind(flow(bundle_total, name="p"), mode=mode)
    exe.run(FRAME)
    assert exe.fallbacks() == {}
    unit = exe.runner.units[exe.plan.calls[0].id]
    assert unit.ragged, "'items' should reach the kernel as flat arrays, not a per-row dispatcher"


def test_a_loop_reading_rows_packs_into_one_kernel():
    def more(mask: int) -> bool:
        return mask <= 0b11

    @step(output="best")
    def keep_best(best: float, bundle_total: float) -> float:
        return bundle_total if bundle_total > best else best

    @step(output="mask")
    def advance(mask: int) -> int:
        return mask + 1

    search = loop(more, flow(bundle_total, keep_best, advance, name="body"),
                 carries=["mask", "best"], max_iterations=4, name="search")
    frame = pl.DataFrame({"items": [[{"price": 2.0}, {"price": 3.0}]], "mask": [1], "best": [0.0]})
    exe = Engine().bind(flow(search, name="p"), mode="fused")
    assert exe.run(frame)["best"].to_list() == [5.0]
    assert "p/search" in exe.runner.packed


def test_a_kernel_that_fails_at_run_time_falls_back_to_the_per_row_dispatcher():
    from numba.core.errors import TypingError

    exe = Engine().bind(flow(bundle_total, name="p"), mode="stepped")
    exe.run(FRAME)
    unit = exe.runner.units[exe.plan.calls[0].id]
    assert unit.ragged, "the setup for this test needs 'items' to have joined the kernel"

    def boom(*a):
        raise TypingError("synthetic failure forcing the retry")

    unit.fn = boom
    assert exe.run(FRAME)["bundle_total"].to_list() == [5.0, 5.0, 0.0]
    assert unit._python, "expected the kernel to have fallen back to one call per row"


def test_rows_runs_compiled_not_interpreted(mode="stepped"):
    from numba.core.dispatcher import Dispatcher

    exe = Engine().bind(flow(bundle_total, name="p"), mode=mode)
    exe.run(FRAME)
    unit = exe.runner.units[exe.plan.calls[0].id]
    assert isinstance(unit.fn, Dispatcher)


class Item2(TypedDict):
    price: float
    weight: float


def test_rows_supports_several_fields():
    def total_weight(items: Columnar[Item2]) -> float:
        total = 0.0
        for j in range(len(items.weight)):
            total += items.weight[j]
        return total

    frame = pl.DataFrame({"items": [[{"price": 1.0, "weight": 4.0}, {"price": 2.0, "weight": 6.0}], []]})
    out = assert_equivalent(flow(total_weight, name="p"), frame)
    assert out["total_weight"].to_list() == [10.0, 0.0]


class Mixed(TypedDict):
    price: float
    qty: int
    taxed: bool


def taxed_total(items: Columnar[Mixed]) -> float:
    total = 0.0
    for j in range(len(items.price)):
        if items.taxed[j]:
            total += items.price[j] * items.qty[j]
    return total


def test_an_item_field_may_be_float_int_or_bool():
    frame = pl.DataFrame({"items": [
        [{"price": 2.0, "qty": 3, "taxed": True}, {"price": 5.0, "qty": 1, "taxed": False}],
        [{"price": 1.5, "qty": 2, "taxed": True}],
        [],
    ]})
    out = assert_equivalent(flow(taxed_total, name="p"), frame)
    assert out["taxed_total"].to_list() == [6.0, 3.0, 0.0]


class Optional(TypedDict):
    price: float
    discount: float | None


def discounted(items: Columnar[Optional]) -> float:
    total = 0.0
    for j in range(len(items.price)):
        d = items.discount[j]
        total += items.price[j] - (0.0 if np.isnan(d) else d)
    return total


def test_a_null_item_field_reads_as_nan_when_it_is_declared_optional():
    frame = pl.DataFrame(
        {"items": [[{"price": 10.0, "discount": 1.0}, {"price": 4.0, "discount": None}], []]},
        schema={"items": pl.List(pl.Struct({"price": pl.Float64, "discount": pl.Float64}))})
    out = assert_equivalent(flow(discounted, name="p"), frame)
    assert out["discounted"].to_list() == [13.0, 0.0]


class Dated(TypedDict):
    price: float
    when: date


def test_an_item_field_of_an_unsupported_type_names_itself_in_the_error():
    def uses(items: Columnar[Dated]) -> float:
        return float(len(items.price))

    with pytest.raises(TypeError, match="field 'when' is <class 'datetime.date'>"):
        Engine().bind(flow(uses, name="p"), mode="stepped").run(
            pl.DataFrame({"items": [[{"price": 1.0, "when": date(2024, 1, 1)}]]}))


class Tagged(TypedDict):
    name: str
    tags: list[str]


def test_a_nested_item_field_is_refused_at_bind_and_points_at_list_dict():
    def uses(items: Columnar[Tagged]) -> float:
        return float(len(items.name))

    with pytest.raises(TypeError, match=r"field 'tags' is list\[str\].*as list\[dict\]"):
        Engine().bind(flow(uses, name="p"), mode="stepped")


@pytest.mark.parametrize("dtype,annotation", ((pl.Int64, int), (pl.Float64, float)))
def test_a_null_item_field_names_the_field_and_the_item_it_is_in(dtype, annotation):
    # Row 2, item 1: a null that used to read as a silent NaN on a float field.
    rows = [[{"n": 1}], [{"n": 2}], [{"n": 3}, {"n": None}]] + [[{"n": 4}]] * ARROW_ROWS
    frame = pl.DataFrame({"items": rows}, schema={"items": pl.List(pl.Struct({"n": dtype}))})
    values, _ = from_series(frame.get_column("items"))
    for source in (frame.get_column("items"), None):
        with pytest.raises(ValueError, match=r"'n' is null in item 1 of row 2"):
            build_rows(values, (("n", annotation),), source, [])


@pytest.mark.parametrize("mode", MODES)
def test_a_null_item_field_raises_in_every_mode_rather_than_summing_as_nan(mode):
    exe = Engine().bind(flow(total, name="p"), mode=mode)
    record = {"items": [{"price": 2.0}, {"price": None}]}
    with pytest.raises(ValueError, match=r"'price' is null in item 1 of row 0"):
        exe.score(record)
    with pytest.raises(ValueError, match=r"'price' is null in item 1 of row 0"):
        exe.run(pl.DataFrame({"items": [record["items"]]},
                             schema={"items": pl.List(pl.Struct({"price": pl.Float64}))}))


def test_a_genuine_nan_in_an_item_field_is_not_mistaken_for_a_null():
    frame = pl.DataFrame({"items": [[{"price": 2.0}, {"price": float("nan")}]]},
                         schema={"items": pl.List(pl.Struct({"price": pl.Float64}))})
    values, _ = from_series(frame.get_column("items"))
    out = build_rows(values, (("price", float),))
    assert np.isnan(out[0].price[1])


NULLS = pl.DataFrame({"items": [[{"price": 2.0}, {"price": 3.0}], None, [], [{"price": 7.0}]]},
                     schema={"items": pl.List(pl.Struct({"price": pl.Float64}))})


def total(items: Columnar[Item]) -> float:
    t = 0.0
    for j in range(len(items.price)):
        t += items.price[j]
    return t


@step(output="picked")
def every_price(items: Columnar[Item]) -> float:
    t = 0.0
    for j in range(len(items.price)):
        t += items.price[j]
    return t


@step(output="picked")
def first_price(items: Columnar[Item]) -> float:
    return items.price[0] if len(items.price) else 0.0


def test_a_null_list_is_required_like_any_other_input():
    from decider.engine.boundary.nulls import MissingInputError

    exe = Engine().bind(flow(total, name="p"), mode="stepped")
    with pytest.raises(MissingInputError, match="'items'"):
        exe.run(NULLS)


def test_missing_as_an_empty_list_reads_a_null_list_as_no_items():
    def filled(items: Columnar[Item] = missing_as([])) -> float:
        t = 0.0
        for j in range(len(items.price)):
            t += items.price[j]
        return t

    out = assert_equivalent(flow(filled, name="p"), NULLS)
    assert out["filled"].to_list() == [5.0, 0.0, 0.0, 7.0]
    absent = assert_equivalent(flow(filled, name="p"), pl.DataFrame({"other": [1.0, 2.0]}))
    assert absent["filled"].to_list() == [0.0, 0.0]


def test_missing_as_a_non_empty_list_says_it_cannot_be_held():
    def filled(items: Columnar[Item] = missing_as([{"price": 1.0}])) -> float:
        return float(len(items.price))

    with pytest.raises(TypeError, match="must be an empty list"):
        Engine().bind(flow(filled, name="p"), mode="stepped").run(NULLS)


def test_an_optional_rows_input_gives_the_step_none_for_a_null_list():
    def maybe(items: Columnar[Item] | None) -> float:
        if items is None:
            return -1.0
        t = 0.0
        for j in range(len(items.price)):
            t += items.price[j]
        return t

    out = assert_equivalent(flow(maybe, name="p"), NULLS)
    assert out["maybe"].to_list() == [5.0, -1.0, 0.0, 7.0]
    # One kernel has one signature, and a namedtuple of array views has no in-band None, so an
    # OPTIONAL Columnar[...] input keeps the per-row dispatcher rather than joining the kernel.
    exe = Engine().bind(flow(maybe, name="p"), mode="fused")
    exe.run(NULLS)
    assert "reads 'items' as Columnar[...]" in exe.fallbacks()["p/maybe"]


def _wide(n: int, items: int = 2) -> pl.DataFrame:
    # Past ARROW_ROWS, so the Arrow read is the path taken.
    return pl.DataFrame(
        {"mask": [0b11] * n,
         "items": [[{"price": float(r * items + k)} for k in range(items)] for r in range(n)]},
        schema={"mask": pl.Int64, "items": pl.List(pl.Struct({"price": pl.Float64}))})


@pytest.mark.parametrize("offset", (0, 1, 7))
def test_the_arrow_read_agrees_with_the_python_read_on_a_sliced_frame(offset):
    frame = _wide(ARROW_ROWS * 2).slice(offset, ARROW_ROWS + 5)
    series = frame.get_column("items")
    values, _ = from_series(series)
    schema = (("price", float),)
    alive: list = []
    arrow = build_rows(values, schema, series, alive)
    assert alive, "the Arrow read was not taken"
    python = build_rows(values, schema)
    assert [r.price.tolist() for r in arrow] == [r.price.tolist() for r in python]


def test_a_null_outside_a_slice_is_not_this_slices_problem():
    rows = [[{"price": None}]] + [[{"price": 1.0}, {"price": 2.0}]] * (ARROW_ROWS + 4)
    frame = pl.DataFrame({"items": rows}, schema={"items": pl.List(pl.Struct({"price": pl.Float64}))})
    series = frame.slice(1, ARROW_ROWS + 3).get_column("items")
    values, _ = from_series(series)
    alive: list = []
    out = build_rows(values, (("price", float),), series, alive)
    assert alive, "the Arrow read was not taken"
    assert [r.price.tolist() for r in out] == [[1.0, 2.0]] * (ARROW_ROWS + 3)


def test_a_batch_past_the_arrow_threshold_agrees_with_every_mode():
    frame = _wide(ARROW_ROWS + 3)
    out = assert_equivalent(flow(bundle_total, name="p"), frame)
    assert out["bundle_total"].to_list() == [
        r[0]["price"] + r[1]["price"] for r in frame.get_column("items").to_list()]


def test_rows_inside_a_branch_reads_only_its_own_rows():
    def big(mask: int) -> bool:
        return mask > 0b01

    pipeline = flow(branch(big, every_price, first_price, modifies=["picked"], name="by"), name="p")
    frame = _wide(ARROW_ROWS + 4).with_columns(
        pl.Series("mask", [0b11, 0b01] * ((ARROW_ROWS + 4) // 2)))
    out = assert_equivalent(pipeline, frame)
    items = frame.get_column("items").to_list()
    assert out["picked"].to_list() == [
        (r[0]["price"] + r[1]["price"]) if k % 2 == 0 else r[0]["price"] for k, r in enumerate(items)]


def test_a_session_override_of_a_rows_input_rebuilds_from_the_new_values():
    exe = Engine().bind(flow(total, name="p"), mode="stepped")
    n = ARROW_ROWS + 2
    s = exe.session(_wide(n))
    s.set("items", [[{"price": 100.0}]] * n)
    s.resume()
    assert s.output()["total"].to_list() == [100.0] * (ARROW_ROWS + 2)


def test_arrow_backed_field_arrays_outlive_the_frame_they_were_read_from():
    frame = _wide(ARROW_ROWS * 4)
    series = frame.get_column("items")
    values, _ = from_series(series)
    alive: list = []
    out = build_rows(values, (("price", float),), series, alive)
    assert alive, "the Arrow read was not taken"
    expected = [r.price.tolist() for r in out]
    # Nothing but `out` itself may be keeping the Arrow buffers alive.
    del frame, series, values, alive
    gc.collect()
    # Fresh allocations reuse released pages, so a dangling view reads rubbish or faults.
    [np.random.default_rng(k).random(200_000) for k in range(8)]
    assert [r.price.tolist() for r in out] == expected


def test_a_plain_list_dict_step_still_runs_in_every_mode():
    def cheapest(items: list) -> float:
        return float(min(i["price"] for i in items))

    frame = pl.DataFrame({"items": [[{"price": 2.0}, {"price": 1.0}], [{"price": 9.0}]]})
    out = assert_equivalent(flow(cheapest, name="p"), frame)
    assert out["cheapest"].to_list() == [1.0, 9.0]


def undeclared(v):
    return v


def bundle_count(items: Columnar[Item]) -> float:
    # No @helper, so the step can't compile and runs in Python; `Columnar[Item]` still holds.
    return float(len(undeclared(items).price))


def test_rows_keeps_its_representation_when_the_step_runs_in_python():
    out = assert_equivalent(flow(bundle_count, name="p"), FRAME)
    assert out["bundle_count"].to_list() == [2.0, 1.0, 0.0]
