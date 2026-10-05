import polars as pl
import pytest

from decider import Executable, flow, param, step
from decider.exceptions import ParamsError
from decider.trace import Kind, TraceSink


def ratio(income: float, debt: float) -> float:
    return debt / income


def affordable(ratio: float, limit: float = param(0.4, ge=0.0, le=1.0)) -> bool:
    return ratio <= limit


pipeline = flow(ratio, affordable, name="afford")
df = pl.DataFrame({"income": [1000.0], "debt": [200.0]})


def test_compile_returns_an_executable_that_scores_a_record():
    exe = pipeline.compile("fused")
    assert isinstance(exe, Executable)
    assert exe.score({"income": 1000.0, "debt": 200.0})["affordable"] is True


def test_run_accepts_a_trace_sink():
    sink = TraceSink()
    pipeline.run(df, trace=sink)
    assert any(e.kind is Kind.STEP for e in sink.events())


def test_validate_accepts_a_good_document():
    result = pipeline.parameters().validate({"afford": {"affordable": {"limit": 0.9}}})
    assert result.errors == ()
    assert result.check() is result


def test_validate_reports_unknown_step_and_bad_value_without_running():
    schema = pipeline.parameters()
    assert "no step with params" in schema.validate({"afford": {"nope": {"limit": 0.9}}}).errors[0]
    result = schema.validate({"afford": {"affordable": {"limit": 7.0}}})
    assert "afford/affordable" in result.errors[0]
    with pytest.raises(ParamsError):
        result.check()


def test_function_step_compiles_and_bind_still_redefaults_params():
    one = step(affordable)
    assert one.compile("fused").score({"ratio": 0.5})["affordable"] is False
    assert one.bind(limit=0.9).compile().score({"ratio": 0.5})["affordable"] is True
