"""`run()` over a frame accepts the object outputs `score()` returns."""
from __future__ import annotations

import polars as pl
import pytest

from decikit import Engine, allow_fallback, flow, step

MODES = ("interpreted", "fused")


@allow_fallback
@step(output="rows")
def mixed_rows(a: float) -> list[dict]:
    return [{"n": 1, "v": int(a)}, {"n": 2, "v": a + 1.5}]


@allow_fallback
@step(output="by_code")
def int_keyed(a: float) -> dict:
    return {1: "x", 2: "y"}


FRAME = pl.DataFrame({"a": [1.0, 2.0]})


def test_a_list_of_dicts_mixing_int_and_float_in_one_field_runs_over_a_frame():
    out = Engine().bind(flow(mixed_rows, name="p"), mode="interpreted").run(FRAME)
    assert out["rows"].to_list() == [[{"n": 1, "v": 1.0}, {"n": 2, "v": 2.5}], [{"n": 1, "v": 2.0}, {"n": 2, "v": 3.5}]]


@pytest.mark.parametrize("mode", MODES)
def test_a_dict_with_int_keys_runs_over_a_frame_with_its_keys_as_text(mode):
    out = Engine().bind(flow(int_keyed, name="p"), mode=mode).run(FRAME)
    assert out["by_code"].to_list() == [{"1": "x", "2": "y"}] * 2
