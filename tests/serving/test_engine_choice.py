"""Serving binds a numba engine only: the polars engine is batch-only and is refused when the handler is made."""
import pytest

from decikit import PolarsEngine, SteppedEngine
from decikit.config import JsonFileStore
from decikit.exceptions import EngineError
from decikit.serving import RequestHandler
from decikit.serving._handler import construct_handler_from_settings


def build():
    from decikit import flow

    def double(x: float) -> float:
        return x * 2

    return flow(double, name="p")


@pytest.fixture
def store(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path))
    store.create_version({"params": {}})
    return store


@pytest.mark.parametrize("engine", ["polars", PolarsEngine()])
def test_a_handler_refuses_the_polars_engine_naming_the_numba_ones(store, engine):
    with pytest.raises(EngineError, match=r"polars.*numba engine.*interpreted.*stepped.*fused"):
        RequestHandler(store, build, engine=engine)


def test_a_handler_accepts_a_numba_engine_tag(store):
    assert RequestHandler(store, build, engine="stepped").engine == SteppedEngine()


def test_the_settings_handler_refuses_the_polars_engine(monkeypatch, tmp_path):
    from decikit._settings import get_settings

    monkeypatch.setenv("DECIKIT_API__ENGINE__TYPE", "polars")
    monkeypatch.setenv("DECIKIT_CONFIG__BASEPATH", str(tmp_path))
    monkeypatch.setenv("DECIKIT_API__PIPELINE", "test_engine_choice:build")
    get_settings.cache_clear()
    try:
        with pytest.raises(EngineError, match="polars"):
            construct_handler_from_settings()
    finally:
        get_settings.cache_clear()
