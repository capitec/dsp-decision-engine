"""`Rows[Item]`: a list column's items as arrays per field, sliced per parent row."""
from __future__ import annotations

from typing import TypedDict

import polars as pl
import pytest

from decider import Engine, Rows, flow
from decider.testing import assert_equivalent

MODES = ("interpreted", "stepped", "fused")


class Item(TypedDict):
    price: float


def bundle_total(mask: int, items: Rows[Item]) -> float:
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
def test_rows_compiles_but_cannot_join_the_shared_array_kernel(mode):
    exe = Engine().bind(flow(bundle_total, name="p"), mode=mode)
    exe.run(FRAME)
    assert "reads 'items' as Rows[...]" in exe.fallbacks()["p/bundle_total"]


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
    def total_weight(items: Rows[Item2]) -> float:
        total = 0.0
        for j in range(len(items.weight)):
            total += items.weight[j]
        return total

    frame = pl.DataFrame({"items": [[{"price": 1.0, "weight": 4.0}, {"price": 2.0, "weight": 6.0}], []]})
    out = assert_equivalent(flow(total_weight, name="p"), frame)
    assert out["total_weight"].to_list() == [10.0, 0.0]
