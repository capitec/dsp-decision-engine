"""Errors from step functions keep their type and name the step; kernels numba can't build run in Python."""
import sys

import polars as pl
import pytest

from decider import branch, flow, frame_step, param, step
from decider.engine import Engine

MODES = ("interpreted", "stepped", "fused")
needs_notes = pytest.mark.skipif(sys.version_info < (3, 11), reason="exception notes are Python 3.11+")


def ratio(income: float, debt: float) -> float:
    if income == 0:
        raise ZeroDivisionError("no income")   # numpy gives inf where numba raises; raise in every mode
    return debt / income


def doubled(ratio: float) -> float:
    return ratio * 2


def is_big(income: float) -> bool:
    return income > 10


@step(output="ratio")
def zero(income: float) -> float:
    return 0.0


FRAME = pl.DataFrame({"income": [4.0, 20.0, 0.0], "debt": [1.0, 2.0, 3.0]})


def notes(err):
    return " ".join(getattr(err.value, "__notes__", []))


@needs_notes
@pytest.mark.parametrize("mode", MODES)
def test_an_error_in_a_step_keeps_its_type_and_names_the_step(mode):
    with pytest.raises(ZeroDivisionError) as err:
        Engine().bind(flow(ratio), mode).run(FRAME)
    assert "in step ratio" in notes(err)


@needs_notes
def test_interpreted_mode_also_names_the_row_counted_in_the_input_frame():
    guarded = branch(is_big, zero, ratio, modifies=["ratio"], name="b")   # rows 0 and 2 take `ratio`
    with pytest.raises(ZeroDivisionError) as err:
        flow(guarded).run(FRAME)
    assert "in step b/ratio, row 2" in notes(err)


@needs_notes
def test_a_fused_kernel_error_names_every_step_it_runs():
    with pytest.raises(ZeroDivisionError) as err:
        Engine().bind(flow(ratio, doubled), "fused").run(FRAME)
    assert "['ratio', 'doubled']" in notes(err) and "mode='stepped'" in notes(err)


@needs_notes
@pytest.mark.parametrize("mode", MODES)
def test_score_errors_name_the_step_too(mode):
    with pytest.raises(ZeroDivisionError) as err:
        Engine().bind(flow(ratio), mode).score({"income": 0.0, "debt": 1.0})
    assert "in step ratio" in notes(err)


@needs_notes
@pytest.mark.parametrize("mode", ("stepped", "fused"))
def test_a_python_fallback_error_names_the_step(mode):
    def imports(income: float) -> float:
        import math  # numba can't compile an import, so this step runs in Python

        return math.log(income)

    with pytest.raises(ValueError) as err:
        Engine().bind(flow(imports), mode).run(FRAME)
    assert "in step imports" in notes(err)


@needs_notes
def test_a_frame_step_error_names_the_step():
    def broken(df: pl.DataFrame) -> pl.DataFrame:
        raise KeyError("nope")

    with pytest.raises(KeyError) as err:
        flow(frame_step(broken)).run(FRAME)
    assert "in frame step broken" in notes(err)


def scaled(ratio: float, table: dict = param({"k": 2.0})) -> float:
    return ratio * table["k"]


@pytest.mark.parametrize("mode", ("stepped", "fused"))
def test_a_kernel_numba_cannot_type_runs_in_python(mode):
    # Each step compiles alone; the dict param is only seen when the kernel launches.
    pipeline = flow(ratio, scaled)
    df = FRAME.head(2)
    expected = pipeline.run(df)
    assert Engine().bind(pipeline, mode).run(df).equals(expected)


@pytest.mark.parametrize("mode", ("stepped", "fused"))
def test_a_step_error_still_propagates_after_its_kernel_fell_back(mode):
    with pytest.raises(ZeroDivisionError):
        Engine().bind(flow(ratio, scaled), mode).run(FRAME)
