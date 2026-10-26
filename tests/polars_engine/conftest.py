import polars as pl
import pytest


@pytest.fixture(autouse=True)
def _polars_defaults():
    yield
    pl.Config.restore_defaults()
