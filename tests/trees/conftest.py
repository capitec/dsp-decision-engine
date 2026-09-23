import polars as pl
import pytest

from decider.engine import Engine

MODES = ("interpreted", "stepped", "fused")


def run_modes(step, df: pl.DataFrame, params=None) -> pl.DataFrame:
    """`step` run over `df` in every mode; the reference walker (interpreted) and the compiled one must agree.

    Every tree test goes through this, so every fixture checks the two walkers against each other.
    """
    outs = {m: Engine().bind(step, mode=m).run(df, params=params) for m in MODES}
    for m in MODES[1:]:
        assert outs[m].equals(outs["interpreted"]), f"{m} disagrees with interpreted:\n{outs[m]}\n{outs['interpreted']}"
        assert outs[m].schema == outs["interpreted"].schema, m
    return outs["interpreted"]


@pytest.fixture
def run():
    return run_modes
