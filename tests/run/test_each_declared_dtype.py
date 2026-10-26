"""Every engine types a declared `each`'s gathered list from the pipeline, never from the items it holds."""
from __future__ import annotations

from datetime import datetime
from typing import TypedDict
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from decikit import PolarsEngine, each, flow, frame_step, missing_as
from decikit.exceptions import EngineError

ENGINES = ("interpreted", "stepped", "fused")


class Line(TypedDict, total=False):
    weight: float
    sku: str
    colour: str


def lines(n: int, flip: bool) -> list[Line]:
    return [{"sku": f"s{k}", "weight": 10.0 * k} if flip else {"weight": 10.0 * k, "sku": f"s{k}"}
            for k in range(n)]


def heavy(weight: float = missing_as(0.0)) -> bool:
    return weight > 15.0


@frame_step(reads=["n"], writes=["rank"])
def rank_n(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("n").rank().alias("rank"))


PIPE = flow(lines, each("lines", flow(heavy, name="line"), name="each_line"), name="p")
GATHERED = pl.List(pl.Struct({"weight": pl.Float64, "sku": pl.String, "colour": pl.String, "heavy": pl.Boolean}))
SCHEMA = {"n": pl.Int64, "flip": pl.Boolean}


def frame(n, flip):
    return pl.DataFrame({"n": n, "flip": flip}, schema=SCHEMA)


def declared(n):
    return [{"weight": 10.0 * k, "sku": f"s{k}", "colour": None, "heavy": 10.0 * k > 15.0} for k in range(n)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("df", [
    frame([], []),
    frame([0, 0], [False, True]),
    frame([2, 1], [False, False]),
    frame([3, 2, 0], [True, False, True]),
], ids=["zero rows", "only empty lists", "no item sets the optional field", "keys in another order"])
def test_declared_each_gathers_into_the_declared_dtype(engine, df):
    out = PIPE.compile(engine).run(df)
    assert out.schema["lines"] == GATHERED
    assert out["lines"].to_list() == [declared(n) for n in df["n"]]


def test_polars_equals_fused_in_memory_streamed_and_sunk(tmp_path):
    df = frame([0, 0, 0, 3, 1, 0, 0, 0, 2, 0], [True, False] * 5)
    want = PIPE.compile("fused").run(df)
    exe = PolarsEngine().bind(PIPE)
    assert_frame_equal(exe.run(df.lazy()).collect(), want)
    pl.Config.set_streaming_chunk_size(3)
    try:
        assert_frame_equal(exe.run(df.lazy()).collect(engine="streaming"), want)
        exe.run(df.lazy()).sink_parquet(tmp_path / "out.parquet")
    finally:
        pl.Config.restore_defaults()
    assert_frame_equal(pl.read_parquet(tmp_path / "out.parquet"), want)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("pipe", [PIPE, flow(PIPE, rank_n, name="ranked")], ids=["record path", "via a frame"])
def test_record_run_returns_the_items_as_built(engine, pipe):
    got = pipe.compile(engine).run({"n": 2, "flip": True})["lines"]
    assert [list(d.items()) for d in got] == [[("sku", "s0"), ("weight", 0.0), ("heavy", False)],
                                              [("sku", "s1"), ("weight", 10.0), ("heavy", False)]]


class Stamped(TypedDict):
    at: datetime


def stamps(n: int) -> list[Stamped]:
    return [{"at": datetime(2020, 1, 1, 12, tzinfo=ZoneInfo("Africa/Johannesburg"))}] * n


STAMPED = flow(stamps, each("stamps", flow(heavy, name="stamp"), name="each_stamp"), name="p")


@pytest.mark.parametrize("engine", ENGINES)
def test_a_zoned_datetime_in_a_naive_declared_field_keeps_its_zone(engine):
    got = STAMPED.compile(engine).run(pl.DataFrame({"n": [1]}))["stamps"].to_list()
    assert got[0][0]["at"] == datetime(2020, 1, 1, 12, tzinfo=ZoneInfo("Africa/Johannesburg"))
    assert got[0][0]["at"].tzinfo is not None


def test_the_polars_engine_refuses_a_zoned_datetime_in_a_naive_declared_field():
    with pytest.raises(EngineError, match="'stamps' came back as .*declared"):
        PolarsEngine().bind(STAMPED).run(pl.DataFrame({"n": [1]}).lazy()).collect()
