"""The polars engine types an `each`'s gathered list from the item's dtype and the body's declared outputs."""
from __future__ import annotations

from typing import Literal, TypedDict

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from decikit import FusedEngine, PolarsEngine, each, flow, missing_as
from decikit._engine.polars.host import PolarsExecutable
from decikit.exceptions import EngineError
from decikit.lab.testing import assert_equivalent


def heavy(weight: float = missing_as(0.0)) -> bool:
    return weight > 20.0


def band(weight: float = missing_as(0.0)) -> Literal["heavy", "light"]:
    return "heavy" if weight > 20.0 else "light"


def tags(weight: float = missing_as(0.0)) -> list[str]:
    return ["big"] if weight > 20.0 else ["small", "cheap"]


def fragile(kg: float = missing_as(0.0)) -> bool:
    return kg < 1.0


PIPE = flow(each("items", flow(heavy, band, tags, each("parts", flow(fragile, name="part"), name="parts"),
                               name="item"), name="items"), name="order")
PART = pl.Struct({"kg": pl.Float64})
ITEM = pl.Struct({"weight": pl.Float64, "meta": pl.Struct({"code": pl.Int64, "note": pl.String}),
                  "parts": pl.List(PART)})
GATHERED = pl.List(pl.Struct({
    "weight": pl.Float64, "meta": pl.Struct({"code": pl.Int64, "note": pl.String}),
    "parts": pl.List(pl.Struct({"kg": pl.Float64, "fragile": pl.Boolean})),
    "heavy": pl.Boolean, "band": pl.String, "tags": pl.List(pl.String)}))


def item(weight, code=1, note="n", parts=((0.5,), (3.0,))):
    return {"weight": weight, "meta": None if code is None else {"code": code, "note": note},
            "parts": [{"kg": kg} for (kg,) in parts]}


def frame(lists):
    return pl.DataFrame({"id": list(range(len(lists))), "items": lists},
                        schema={"id": pl.Int64, "items": pl.List(ITEM)})


MIXED = frame([[item(30.0), item(None, code=None)], [], None, [item(5.0, note=None, parts=())]] * 5)


def host(df, pipe=PIPE):
    return PolarsEngine().bind(pipe).run(df.lazy())


def streamed(lf, chunk=3):
    pl.Config.set_streaming_chunk_size(chunk)
    return lf.collect(engine="streaming")


def test_each_with_declared_outputs_equals_fused_in_memory_streamed_and_sunk(tmp_path):
    want = FusedEngine().bind(PIPE).run(MIXED)
    assert want.schema["items"] == GATHERED
    assert_frame_equal(host(MIXED).collect(), want)
    assert_frame_equal(streamed(host(MIXED)), want)
    MIXED.write_parquet(tmp_path / "in.parquet")
    PolarsEngine().bind(PIPE).run(pl.scan_parquet(tmp_path / "in.parquet")).sink_parquet(tmp_path / "out.parquet")
    assert_frame_equal(pl.read_parquet(tmp_path / "out.parquet"), want)


def test_batch_of_only_empty_lists_keeps_the_gathered_dtype():
    df = frame([[], [], None])
    got = host(df).collect()
    assert got.schema["items"] == GATHERED
    assert got["items"].to_list() == [[], [], []]


def test_zero_row_frame_has_the_gathered_dtype():
    lf = host(frame([]))
    assert lf.collect_schema()["items"] == GATHERED
    assert lf.collect().schema["items"] == GATHERED


def test_streamed_batches_of_only_empty_lists_share_the_dtype_of_full_ones(tmp_path):
    df = frame([[]] * 7 + [[item(30.0)]] * 2 + [[]] * 6 + [[item(1.0, code=None)]])
    want = FusedEngine().bind(PIPE).run(df)
    assert_frame_equal(streamed(host(df)), want)
    host(df).sink_parquet(tmp_path / "out.parquet")
    assert_frame_equal(pl.read_parquet(tmp_path / "out.parquet"), want)


class Line(TypedDict):
    weight: float
    sku: str


def lines(n: int) -> list[Line]:
    return [{"weight": 10.0 * k, "sku": f"s{k}"} for k in range(n)]


def test_item_column_a_step_declares_is_typed_from_its_declaration():
    pipe = flow(lines, each("lines", flow(heavy, name="line"), name="each_line"), name="p")
    df = pl.DataFrame({"n": [0, 3, 1, 0]})
    want = FusedEngine().bind(pipe).run(df)
    assert_frame_equal(streamed(host(df, pipe)), want)
    assert host(df.clear(), pipe).collect().schema["lines"] == pl.List(
        pl.Struct({"weight": pl.Float64, "sku": pl.String, "heavy": pl.Boolean}))


def test_undeclared_body_output_is_refused_naming_the_each_and_the_output():
    def details(weight: float = missing_as(0.0)) -> dict:
        return {"w": weight}

    pipe = flow(each("items", flow(details, name="item"), name="items"), name="order")
    with pytest.raises(EngineError, match=r"order/items.*'details'.*order/items/item/details.*Annotate") as e:
        host(MIXED, pipe)
    assert "numba engine" in str(e.value)


def test_undeclared_item_column_is_refused_naming_the_step_that_writes_it():
    def loose(n: int) -> list[dict]:
        return [{"weight": 1.0}] * n

    pipe = flow(loose, each("loose", flow(heavy, name="line"), name="each_line"), name="p")
    with pytest.raises(EngineError, match=r"p/each_line.*'loose'.*p/loose.*list\[dict\].*Annotate"):
        host(pl.DataFrame({"n": [1, 2]}), pipe)


def test_assert_equivalent_accepts_polars_typing_a_field_fused_leaves_null(monkeypatch):
    df = frame([[item(None, code=None, parts=())]])
    assert FusedEngine().bind(PIPE).run(df).schema["items"] != GATHERED
    assert assert_equivalent(PIPE, df)["items"].to_list() == FusedEngine().bind(PIPE).run(df)["items"].to_list()
    run = PolarsExecutable.run
    monkeypatch.setattr(PolarsExecutable, "run", lambda self, data, params=None, trace=None:
                        run(self, data, params).with_columns(pl.col("id").cast(pl.Float64)))
    with pytest.raises(AssertionError, match="polars engine differs"):
        assert_equivalent(PIPE, df, engines=("fused", "polars"))
