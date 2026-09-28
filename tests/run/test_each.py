"""`each`: run a child flow on every element of a list column, writing the enriched list back."""
from __future__ import annotations

from typing import TypedDict

import polars as pl
import pytest

from decider import Columnar, EachMode, Engine, each, flow, missing_as, param
from decider.testing import assert_equivalent

MODES = ("interpreted", "stepped", "fused")


def heavy(weight: float = missing_as(0.0), heavy_kg: float = param(20.0)) -> bool:
    return weight > heavy_kg


def pipeline(mode: EachMode = EachMode.PER_ROW):
    return flow(each("items", flow(heavy, name="item"), name="items", execution_mode=mode), name="order")


FRAME = pl.DataFrame({"items": [[{"weight": 5.0}, {"weight": 25.0}], [], [{"weight": 40.0}]]})


def test_each_adds_child_outputs_as_fields():
    out = assert_equivalent(pipeline(), FRAME)
    assert out["items"].to_list() == [
        [{"weight": 5.0, "heavy": False}, {"weight": 25.0, "heavy": True}],
        [],
        [{"weight": 40.0, "heavy": True}],
    ]


@pytest.mark.parametrize("mode", MODES)
def test_each_batch_agrees_with_per_row(mode):
    exe = Engine().bind(pipeline(EachMode.BATCH), mode=mode)
    out = exe.run(FRAME)
    assert out["items"].to_list() == [
        [{"weight": 5.0, "heavy": False}, {"weight": 25.0, "heavy": True}],
        [],
        [{"weight": 40.0, "heavy": True}],
    ]


def test_a_null_list_reads_as_no_items():
    frame = pl.DataFrame({"items": [[{"weight": 5.0}], None]})
    out = assert_equivalent(pipeline(), frame)
    assert out["items"].to_list() == [[{"weight": 5.0, "heavy": False}], []]


@pytest.mark.parametrize("record", ({"items": []}, {"items": None}))
def test_batch_handles_empty_and_null_lists_on_a_single_record(record):
    exe = Engine().bind(pipeline(EachMode.BATCH), mode="fused")
    assert exe.score(record)["items"] == []


def test_the_childs_params_are_tunable_through_the_parent_document():
    exe = Engine().bind(pipeline(), mode="interpreted")
    record = {"items": [{"weight": 25.0}]}
    assert exe.score(record)["items"] == [{"weight": 25.0, "heavy": True}]
    params = {"order": {"items": {"heavy_kg": 30.0}}}
    assert exe.score(record, params=params)["items"] == [{"weight": 25.0, "heavy": False}]
    assert exe.plan is not None  # smoke: the hoisted param is part of the schema


def test_a_missing_item_field_uses_the_childs_fill():
    frame = pl.DataFrame({"items": [[{"weight": None}]]},
                         schema={"items": pl.List(pl.Struct({"weight": pl.Float64}))})
    out = assert_equivalent(pipeline(), frame)
    assert out["items"].to_list() == [[{"weight": None, "heavy": False}]]


class Item(TypedDict):
    weight: float
    heavy: bool


def bundle_total(items: Columnar[Item]) -> float:
    total = 0.0
    for j in range(len(items.weight)):
        if items.heavy[j]:
            total += items.weight[j]
    return total


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("each_mode", EachMode)
def test_a_parent_step_reads_the_enriched_list_as_columnar(mode, each_mode):
    p = flow(each("items", flow(heavy, name="item"), name="items", execution_mode=each_mode),
             bundle_total, name="order")
    exe = Engine().bind(p, mode=mode)
    assert exe.run(FRAME)["bundle_total"].to_list() == [25.0, 0.0, 40.0]


def test_batch_mode_does_not_inject_parent_frame_columns_into_items():
    frame = pl.DataFrame({"order_id": [1, 2], "items": [[{"weight": 5.0}, {"weight": 25.0}], []]})
    exe = Engine().bind(pipeline(EachMode.BATCH), mode="fused")
    result = exe.run(frame)["items"].to_list()
    assert result == [[{"weight": 5.0, "heavy": False}, {"weight": 25.0, "heavy": True}], []]


def test_a_parent_step_reads_the_enriched_list_as_columnar_past_the_arrow_threshold():
    from decider.engine.compile.rows import ARROW_ROWS

    frame = pl.DataFrame({"items": [[{"weight": float(i % 3)}, {"weight": 30.0}] for i in range(ARROW_ROWS + 4)]})
    p = flow(each("items", flow(heavy, name="item"), name="items", execution_mode=EachMode.BATCH),
             bundle_total, name="order")
    exe = Engine().bind(p, mode="fused")
    assert exe.run(frame)["bundle_total"].to_list() == [30.0] * (ARROW_ROWS + 4)
