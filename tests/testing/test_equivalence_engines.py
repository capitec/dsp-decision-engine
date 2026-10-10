"""assert_equivalent across engines: polars by default, refusals skipped, frame steps compared one row at a time."""
import polars as pl
import pytest

from decikit import flow, frame_step
from decikit._engine.polars.host import PolarsExecutable
from decikit.exceptions import EngineError
from decikit.lab.testing import assert_equivalent


def scaled(a: float) -> float:
    return a * 2


def details(a: float) -> dict:
    return {"a": a}


@frame_step(reads=["a"], writes=["rank"])
def rank_a(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("a").rank().alias("rank"))


FRAME = pl.DataFrame({"a": [3.0, 1.0, 2.0]})


def test_the_polars_engine_runs_by_default_and_a_difference_fails(monkeypatch):
    run = PolarsExecutable.run

    def off_by_one(self, data, params=None, trace=None):
        out = run(self, data, params, trace)
        return out.with_columns(pl.col("scaled") + 1) if isinstance(out, pl.LazyFrame) else out

    monkeypatch.setattr(PolarsExecutable, "run", off_by_one)
    with pytest.raises(AssertionError, match=r"run\(\) on the polars engine differs from interpreted"):
        assert_equivalent(flow(scaled), FRAME)


def test_raising_only_on_a_later_engine_is_a_divergence(monkeypatch):
    def broken(self, data, params=None, trace=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(PolarsExecutable, "run", broken)
    with pytest.raises(AssertionError, match=r"run\(\) raised RuntimeError\('boom'\) on the polars engine but not on"):
        assert_equivalent(flow(scaled), FRAME)


def test_an_engine_that_refuses_the_pipeline_is_skipped_with_a_warning():
    with pytest.warns(UserWarning, match="skipped the polars engine, which refuses this pipeline.*'details'"):
        out = assert_equivalent(flow(details), FRAME)
    assert out["details"].to_list() == [{"a": 3.0}, {"a": 1.0}, {"a": 2.0}]


def test_a_refusal_by_the_first_engine_propagates():
    with pytest.raises(EngineError, match="'details'"):
        assert_equivalent(flow(details), FRAME, engines=("polars", "fused"))


def test_a_record_is_compared_with_a_one_row_frame_when_the_pipeline_has_a_frame_step():
    out = assert_equivalent(flow(rank_a), FRAME)   # a record alone ranks 1.0; in the frame it ranks 3.0
    assert out["rank"].to_list() == [3.0, 1.0, 2.0]
