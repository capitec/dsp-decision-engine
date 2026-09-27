"""A `str` field of a `Columnar[Item]` item: a span into the list's own string buffers."""
from __future__ import annotations

import gc
from typing import TypedDict

import numpy as np
import polars as pl
import pytest

from decider import Engine, Columnar, flow, missing_as, param
from decider.engine.compile.rows import ARROW_ROWS, build_rows
from decider.testing import assert_equivalent

MODES = ("interpreted", "stepped", "fused")
COMPILED = ("stepped", "fused")
# ASCII, empty, 2-byte, 3-byte and 4-byte values, in one row so one call sees them all.
VALUES = ["private", "", "privé", "ééé", "café bar", "subénd", "\U0001f600x"]
FRAME = pl.DataFrame({"items": [[{"label": v} for v in VALUES]]})
WANT = "privé"


class Labelled(TypedDict):
    label: str


class Item(TypedDict):
    el_1: int
    el_2: str


class Optional(TypedDict):
    el_1: int
    el_2: str | None


def eq_literal(items: Columnar[Labelled]) -> int:
    n = 0
    for j in range(len(items.label)):
        if items.label[j] == WANT:
            n += 1
    return n


def code_points(items: Columnar[Labelled]) -> int:
    n = 0
    for j in range(len(items.label)):
        n += len(items.label[j])
    return n


def byte_lengths(items: Columnar[Labelled]) -> int:
    n = 0
    for j in range(len(items.label)):
        n += items.label[j][1]
    return n


def starts(items: Columnar[Labelled]) -> int:
    n = 0
    for j in range(len(items.label)):
        if items.label[j].startswith("priv"):
            n += 1
    return n


def ends(items: Columnar[Labelled]) -> int:
    n = 0
    for j in range(len(items.label)):
        if items.label[j].endswith("énd"):
            n += 1
    return n


def holds(items: Columnar[Labelled]) -> int:
    n = 0
    for j in range(len(items.label)):
        if "afé" in items.label[j]:
            n += 1
    return n


def iterated(items: Columnar[Labelled]) -> int:
    n = 0
    for label in items.label:
        if label == WANT:
            n += 1
    return n


CPYTHON = {
    "eq_literal": sum(v == WANT for v in VALUES),
    "code_points": sum(len(v) for v in VALUES),
    "byte_lengths": sum(len(v.encode()) for v in VALUES),
    "starts": sum(v.startswith("priv") for v in VALUES),
    "ends": sum(v.endswith("énd") for v in VALUES),
    "holds": sum("afé" in v for v in VALUES),
    "iterated": sum(v == WANT for v in VALUES),
}
STEPS = [eq_literal, code_points, byte_lengths, starts, ends, holds, iterated]


@pytest.mark.parametrize("fn", STEPS)
def test_a_str_field_answers_as_cpython_does_in_every_mode(fn):
    out = assert_equivalent(flow(fn, name="p"), FRAME)
    assert out[fn.__name__].to_list() == [CPYTHON[fn.__name__]]


@pytest.mark.parametrize("fn", STEPS)
def test_the_arrow_read_answers_the_same_as_the_python_read(fn):
    # Above ARROW_ROWS the field is read straight out of the list's own string buffers.
    frame = pl.DataFrame({"items": [[{"label": v} for v in VALUES]] * (ARROW_ROWS + 4)})
    exe = Engine().bind(flow(fn, name="p"), mode="fused")
    assert set(exe.run(frame)[fn.__name__].to_list()) == {CPYTHON[fn.__name__]}


def find_best_item(items: Columnar[Item]) -> int:
    for j in range(len(items.el_1)):
        if items.el_1[j] == 400 and items.el_2[j] == "snoop":
            return j
    return -1


ROWS = [[{"el_1": 1, "el_2": "dogg"}, {"el_1": 400, "el_2": "snoop"}],
        [{"el_1": 400, "el_2": "dogg"}],
        [{"el_1": 400, "el_2": "privé"}, {"el_1": 400, "el_2": "snoop"}],
        []]
FOUND = [1, -1, 1, -1]


def test_a_str_field_beside_a_number_finds_the_item_in_every_mode():
    assert assert_equivalent(flow(find_best_item, name="p"), pl.DataFrame({"items": ROWS})
                             )["find_best_item"].to_list() == FOUND


@pytest.mark.parametrize("mode", MODES)
def test_score_reads_one_record_through_the_same_operations(mode):
    exe = Engine().bind(flow(find_best_item, name="p"), mode=mode)
    assert [exe.score({"items": row})["find_best_item"] for row in ROWS] == FOUND


def test_each_row_reads_only_its_own_items_on_the_arrow_path():
    frame = pl.DataFrame({"items": ROWS * (ARROW_ROWS // 2)})
    exe = Engine().bind(flow(find_best_item, name="p"), mode="fused")
    assert exe.run(frame)["find_best_item"].to_list() == FOUND * (ARROW_ROWS // 2)
    # A slice carries its offset on the parent list only; the spans must still be windowed to it.
    assert exe.run(frame.slice(1, ARROW_ROWS))["find_best_item"].to_list() == (
        (FOUND[1:] + FOUND * (ARROW_ROWS // 2))[:ARROW_ROWS])


class Pair(TypedDict):
    left: str
    right: str


def eq_global(items: Columnar[Pair]) -> int:
    n = 0
    for j in range(len(items.left)):
        if items.left[j] == WANT:
            n += 1
    return n


def eq_other_field(items: Columnar[Pair]) -> int:
    n = 0
    for j in range(len(items.left)):
        if items.left[j] == items.right[j]:
            n += 1
    return n


def eq_param(items: Columnar[Pair], want: str = param(WANT)) -> int:
    n = 0
    for j in range(len(items.left)):
        if items.left[j] == want:
            n += 1
    return n


PAIRS = pl.DataFrame({"items": [[{"left": WANT, "right": WANT}, {"left": "other", "right": WANT}]]})


@pytest.mark.parametrize("fn", (eq_global, eq_other_field, eq_param))
def test_a_str_field_compares_against_a_constant_a_param_and_another_field(fn):
    assert assert_equivalent(flow(fn, name="p"), PAIRS)[fn.__name__].to_list() == [1]


def test_a_str_field_compared_against_a_param_runs_in_the_kernel():
    # `_probe_signature` typed `want` from `SPAN in ins`, which only sees a top-level `bytes`
    # input; a `str` field nested inside a `Columnar[...]` input's schema was invisible to it, so
    # `want` was typed as a Raw[str] code and the step fell back to Python.
    exe = Engine().bind(flow(eq_param, name="p"), mode="fused")
    exe.run(PAIRS)
    assert exe.fallbacks() == {}


def optional_nulls(items: Columnar[Optional]) -> int:
    n = 0
    for j in range(len(items.el_1)):
        # A null is not a string: it equals nothing, holds nothing and reads as empty.
        if items.el_2[j] == "" or items.el_2[j].startswith("") or len(items.el_2[j]) > 0:
            n += 1
    return n


NULLABLE = pl.DataFrame({"items": [[{"el_1": 1, "el_2": None}, {"el_1": 2, "el_2": ""},
                                    {"el_1": 3, "el_2": "x"}]]},
                        schema={"items": pl.List(pl.Struct({"el_1": pl.Int64, "el_2": pl.String}))})


def test_an_optional_str_field_reads_a_null_as_a_null_span():
    assert assert_equivalent(flow(optional_nulls, name="p"), NULLABLE)["optional_nulls"].to_list() == [2]


def null_test(items: Columnar[Optional]) -> int:
    n = 0
    for j in range(len(items.el_1)):
        if items.el_2[j][1] < 0:
            n += 1
    return n


def test_the_byte_length_is_the_null_test_in_every_mode():
    assert assert_equivalent(flow(null_test, name="p"), NULLABLE)["null_test"].to_list() == [1]


def required_str(items: Columnar[Item]) -> int:
    return len(items.el_1)


@pytest.mark.parametrize("n", (1, ARROW_ROWS + 4))
def test_a_null_in_a_required_str_field_names_the_item_it_is_in(n):
    rows = [[{"el_1": 1, "el_2": "a"}, {"el_1": 2, "el_2": None}]] * n
    frame = pl.DataFrame({"items": rows},
                         schema={"items": pl.List(pl.Struct({"el_1": pl.Int64, "el_2": pl.String}))})
    with pytest.raises(ValueError, match="'el_2' is null in item 1 of row 0.*`str | None`"):
        Engine().bind(flow(required_str, name="p"), mode="stepped").run(frame)


def test_an_int_field_still_refuses_to_be_optional():
    with pytest.raises(TypeError, match=r"only `float \| None` and `str \| None` do"):
        build_rows(np.array([None], object), (("n", int | None),))


def n_snoop(items: Columnar[Item] = missing_as([])) -> int:
    n = 0
    for j in range(len(items.el_1)):
        if items.el_2[j] == "snoop":
            n += 1
    return n


@pytest.mark.parametrize("mult", (1, ARROW_ROWS))
def test_a_null_or_empty_list_reads_no_items_beside_a_str_field(mult):
    rows = [[{"el_1": 1, "el_2": "snoop"}, {"el_1": 2, "el_2": "x"}], None, [],
            [{"el_1": 3, "el_2": "snoop"}]]
    frame = pl.DataFrame({"items": rows * mult},
                         schema={"items": pl.List(pl.Struct({"el_1": pl.Int64, "el_2": pl.String}))})
    exe = Engine().bind(flow(n_snoop, name="p"), mode="fused")
    assert exe.run(frame)["n_snoop"].to_list() == [1, 0, 0, 1] * mult


def test_span_fields_outlive_the_frame_they_were_read_from():
    n = ARROW_ROWS + 4
    frame = pl.DataFrame({"items": [[{"label": "a" * 40}, {"label": "privé" * 8}]] * n})
    series = frame.get_column("items")
    values = np.empty(n, object)
    for i, row in enumerate(series.to_list()):
        values[i] = row
    rows = build_rows(values, (("label", str),), series, [])
    del frame, series, values
    gc.collect()
    # Reuse the released pages before reading the spans back.
    ballast = [np.full(200_000, 1.5) for _ in range(8)]
    assert len(rows[0].label[0]) == 40
    assert rows[n - 1].label[1] == "privé" * 8
    del ballast
