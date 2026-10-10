"""Step.run on a dict, a DataFrame and a LazyFrame, on every engine; a LazyFrame alone picks polars."""
import polars as pl
import pytest

import decikit
from decikit import FusedEngine, flow, param
from decikit.exceptions import EngineError

ENGINES = ("interpreted", "stepped", "fused", "polars")


def ratio(income: float, debt: float) -> float:
    return debt / income


def affordable(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit


def details(ratio: float) -> dict:
    return {"ratio": ratio}


PIPE = flow(ratio, affordable, name="p")
DF = pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, 400.0]})
WANT = FusedEngine().bind(PIPE).run(DF)


@pytest.mark.parametrize("engine", ENGINES)
def test_step_run_takes_a_dict_a_dataframe_and_a_lazyframe_on_every_engine(engine):
    assert PIPE.run(DF, engine=engine).equals(WANT)
    assert PIPE.run({"income": 1000.0, "debt": 200.0}, engine=engine) == WANT.row(0, named=True)
    out = PIPE.run(DF.lazy(), engine=engine)
    # A numba engine has no lazy plan, so it collects; only polars hands back a plan.
    assert isinstance(out, pl.LazyFrame) == (engine == "polars")
    assert (out.collect() if engine == "polars" else out).equals(WANT)


def test_a_lazyframe_runs_on_the_polars_engine_whatever_the_default():
    with decikit.engine("interpreted"):
        out = PIPE.run(DF.lazy())
        assert isinstance(PIPE.run(DF), pl.DataFrame)
    assert isinstance(out, pl.LazyFrame) and out.collect().equals(WANT)


@pytest.mark.parametrize("engine", ENGINES)
def test_an_input_that_is_no_record_or_frame_is_refused_on_every_engine(engine):
    with pytest.raises(TypeError, match="dict record, a polars DataFrame or LazyFrame, not list"):
        PIPE.run([1.0, 2.0], engine=engine)


def test_what_polars_refuses_on_a_lazyframe_runs_when_a_numba_engine_is_named():
    pipe = flow(ratio, details, name="d")
    with pytest.raises(EngineError, match="'details'.*TypedDict"):
        pipe.run(DF.lazy())
    with pytest.raises(EngineError, match="no trace"):
        PIPE.run(DF.lazy(), trace=object())
    assert pipe.run(DF.lazy(), engine="fused")["details"].to_list() == [{"ratio": 0.2}, {"ratio": 0.8}]
