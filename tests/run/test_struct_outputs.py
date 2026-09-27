"""A step returning a struct or a list of records, declared `Struct[Item]` / `Columnar[Item]`."""
from __future__ import annotations

from typing import TypedDict

import polars as pl

from decider import Columnar, Engine, Struct, flow, step


class Product(TypedDict):
    rate: float
    code: int


@step(output="offer")
def best(rate: float, code: int) -> Struct[Product]:
    return {"rate": rate, "code": code}


@step(output="offers")
def ladder(code: int) -> Columnar[Product]:
    return [{"rate": 0.1, "code": code}, {"rate": 0.2, "code": code + 1}]


def test_a_struct_output_keeps_its_shape_on_empty_rows():
    exe = Engine().bind(flow(best, name="order"), mode="fused")
    out = exe.run(pl.DataFrame({"rate": [0.1, 0.2], "code": [1, 2]}))
    assert out["offer"].dtype == pl.Struct({"rate": pl.Float64, "code": pl.Int64})
    assert out["offer"].to_list() == [{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}]
    empty = exe.run(pl.DataFrame({"rate": [], "code": []}))
    assert empty["offer"].dtype == pl.Struct({"rate": pl.Float64, "code": pl.Int64})


def test_a_columnar_output_keeps_its_shape_on_empty_rows():
    exe = Engine().bind(flow(ladder, name="order"), mode="fused")
    out = exe.run(pl.DataFrame({"code": [1]}))
    assert out["offers"].dtype == pl.List(pl.Struct({"rate": pl.Float64, "code": pl.Int64}))
    assert out["offers"].to_list() == [[{"rate": 0.1, "code": 1}, {"rate": 0.2, "code": 2}]]
    empty = exe.run(pl.DataFrame({"code": []}))
    assert empty["offers"].dtype == pl.List(pl.Struct({"rate": pl.Float64, "code": pl.Int64}))
