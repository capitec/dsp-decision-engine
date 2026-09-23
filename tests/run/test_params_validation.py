"""Eager and lazy params validation: per node, cached per document, reported."""
from __future__ import annotations

import polars as pl
import pytest

from decider import branch, param, step
from decider.engine import Engine
from decider.engine.params import ParamsError


def is_big(x: float) -> bool:
    return x > 100


@step(output="y")
def big(x: float, k: float = param(2.0, ge=0)) -> float:
    return x * k


@step(output="y")
def small(x: float, k: float = param(1.0, ge=0)) -> float:
    return x * k


pipeline = branch(is_big, big, small, modifies=["y"], name="by")
SMALL = pl.DataFrame({"x": [1.0, 2.0]})
MIXED = pl.DataFrame({"x": [1.0, 200.0, 300.0]})
BAD_BIG = {"by": {"big": {"k": -1.0}}}


def test_eager_raises_for_an_invalid_param_in_an_arm_no_row_takes(bind):
    with pytest.raises(ParamsError, match=r"by/big: param 'k'.*-1\.0"):
        bind(pipeline, params_validation="eager").run(SMALL, params=BAD_BIG)


def test_eager_is_the_default(bind):
    with pytest.raises(ParamsError):
        bind(pipeline).run(SMALL, params=BAD_BIG)


def test_lazy_runs_when_the_invalid_node_is_never_reached(bind):
    exe = bind(pipeline, params_validation="lazy")
    assert exe.run(SMALL, params=BAD_BIG)["y"].to_list() == [1.0, 2.0]
    assert exe.report.validated == ["by/small"]
    assert exe.report.invalid == []


def test_lazy_raises_when_a_row_reaches_the_invalid_node_naming_its_rows(bind):
    exe = bind(pipeline, params_validation="lazy")
    with pytest.raises(ParamsError, match=r"\(affects 2 rows\)"):
        exe.run(MIXED, params=BAD_BIG)
    assert exe.report.invalid == ["by/big"]


def test_lazy_validates_only_nodes_not_seen_before_for_that_document(bind):
    exe = bind(pipeline, params_validation="lazy")
    exe.run(SMALL)
    assert exe.report.validated == ["by/small"]
    exe.run(MIXED)
    assert exe.report.validated == ["by/big"]
    exe.run(MIXED)
    assert exe.report.validated == []
    exe.run(MIXED, params={"by": {"small": {"k": 3.0}}})
    assert exe.report.validated == ["by/big", "by/small"]


def test_eager_validates_every_node_once_per_document(bind):
    exe = bind(pipeline)
    exe.run(SMALL)
    assert sorted(exe.report.validated) == ["by/big", "by/small"]
    exe.run(SMALL)
    assert exe.report.validated == []


def test_eager_keeps_raising_for_a_document_it_found_invalid(bind):
    exe = bind(pipeline)
    for _ in range(2):
        with pytest.raises(ParamsError):
            exe.run(SMALL, params=BAD_BIG)


def test_a_missing_required_param_is_invalid(bind):
    @step(output="z")
    def needs(x: float, rate: float = param(required=True)) -> float:
        return x * rate

    with pytest.raises(ParamsError, match="needs: param 'rate' is required but missing"):
        bind(needs).run(SMALL)
    assert bind(needs).run(SMALL, params={"needs": {"rate": 2.0}})["z"].to_list() == [2.0, 4.0]


def warned(x: float, k: float = param(2.0, ge=0, on_invalid="warn")) -> float:
    return x * k


def defaulted(x: float, k: float = param(2.0, ge=0, on_invalid="default")) -> float:
    return x * k


@pytest.mark.parametrize("validation", ["eager", "lazy"])
def test_on_invalid_warn_uses_the_default_and_reports_a_warning(validation, bind):
    exe = bind(step(warned), params_validation=validation)
    assert exe.run(SMALL, params={"warned": {"k": -1.0}})["warned"].to_list() == [2.0, 4.0]
    assert len(exe.report.warnings) == 1 and "warned: param 'k'" in exe.report.warnings[0]


@pytest.mark.parametrize("validation", ["eager", "lazy"])
def test_on_invalid_default_uses_the_default_silently(validation, bind):
    exe = bind(step(defaulted), params_validation=validation)
    assert exe.run(SMALL, params={"defaulted": {"k": -1.0}})["defaulted"].to_list() == [2.0, 4.0]
    assert exe.report.warnings == []


def test_lazy_warns_only_for_nodes_that_run(bind):
    @step(output="y")
    def warn_big(x: float, k: float = param(2.0, ge=0, on_invalid="warn")) -> float:
        return x * k

    exe = bind(branch(is_big, warn_big, small, modifies=["y"], name="by"), params_validation="lazy")
    exe.run(SMALL, params={"by": {"warn_big": {"k": -1.0}}})
    assert exe.report.warnings == []


def test_a_params_document_that_is_not_a_mapping_is_an_error(bind):
    with pytest.raises(ParamsError, match="mapping"):
        bind(pipeline).run(SMALL, params=[1, 2])


def test_an_unknown_params_validation_mode_is_an_error(bind):
    with pytest.raises(ValueError, match="'eager' or 'lazy'"):
        Engine(params_validation="sometimes")
