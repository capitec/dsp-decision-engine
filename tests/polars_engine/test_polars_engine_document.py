import polars as pl
import pytest
from pydantic import ValidationError

from decikit import Engine, FusedEngine, PolarsEngine, flow
from decikit.exceptions import EngineError


def ratio(income: float, debt: float) -> float:
    return debt / income


def label(ratio: float) -> str:
    return "low" if ratio < 0.3 else "high"


DF = pl.DataFrame({"income": [1000.0, 500.0], "debt": [200.0, 400.0]})


def test_compile_polars_binds_the_polars_engine():
    exe = flow(ratio, name="p").compile("polars")
    assert Engine.resolve("polars") is PolarsEngine
    assert exe.engine == PolarsEngine()
    assert exe.run(DF.lazy()).collect()["ratio"].to_list() == [0.2, 0.8]


def test_polars_engine_is_a_settings_document_that_round_trips():
    engine = PolarsEngine(segments=FusedEngine(strict_compile=True))
    assert Engine.resolve(engine.type).model_validate(engine.model_dump()) == engine


def test_strict_compile_on_the_segments_refuses_a_python_step():
    exe = PolarsEngine(segments=FusedEngine(strict_compile=True)).bind(flow(ratio, label, name="p"))
    with pytest.raises(EngineError, match="p/label"):
        exe.run(DF.lazy())


def test_params_validation_is_one_setting_shared_with_the_segments():
    with pytest.raises(ValidationError, match="set both the same"):
        PolarsEngine(params_validation="lazy")
    with pytest.raises(ValidationError, match="set both the same"):
        PolarsEngine(segments=FusedEngine(params_validation="lazy"))
    lazy = PolarsEngine(params_validation="lazy", segments=FusedEngine(params_validation="lazy"))
    assert lazy.bind(flow(ratio, name="p")).full.lazy


def test_a_session_on_the_polars_engine_is_refused():
    with pytest.raises(EngineError, match="no debug session; open one on the numba engine"):
        flow(ratio, name="p").session(DF, engine="polars")
