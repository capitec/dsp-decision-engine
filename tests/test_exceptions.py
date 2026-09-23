"""Library errors share `DeciderError` and keep their builtin base, so either `except` catches them."""
from __future__ import annotations

import polars as pl
import pytest

import decider
from decider import branch, dag, flow, param, step
from decider.engine import Engine, to_ir
from decider.engine.boundary._arrow.plan import ArrowKindError as ArrowKindFromPlan
from decider.engine.boundary.dtypes import NeedsKernelSplit as NeedsKernelSplitFromDtypes
from decider.engine.boundary.nulls import MissingInputError as MissingFromNulls
from decider.engine.params import ParamsError as ParamsFromParams
from decider.exceptions import (
    ArrowImportError, ArrowKindError, DeciderError, EngineError, ExprError, IRError, MissingInputError,
    NeedsKernelSplit, ParamsError, RegistryError, WiringError,
)
from decider.steps import ConfigurableStep
from decider.steps.expr import ExprError as ExprFromExpr, parse


@pytest.mark.parametrize("error, builtin", [
    (WiringError, ValueError), (IRError, TypeError), (ParamsError, ValueError), (MissingInputError, ValueError),
    (ArrowKindError, TypeError), (NeedsKernelSplit, TypeError), (ArrowImportError, RuntimeError),
    (EngineError, ValueError), (RegistryError, LookupError), (ExprError, ValueError),
])
def test_each_library_error_is_a_decider_error_and_its_builtin(error, builtin):
    assert issubclass(error, DeciderError) and issubclass(error, builtin)


def test_the_old_import_locations_name_the_same_classes():
    assert ArrowKindFromPlan is ArrowKindError and NeedsKernelSplitFromDtypes is NeedsKernelSplit
    assert MissingFromNulls is MissingInputError and ParamsFromParams is ParamsError
    assert ExprFromExpr is ExprError


def ratio(income: float, debt: float) -> float:
    return debt / income


def test_build_and_run_failures_are_caught_by_decider_error():
    df = pl.DataFrame({"income": [1.0]})
    with pytest.raises(DeciderError):          # missing input column
        flow(ratio).run(df)
    with pytest.raises(DeciderError):          # invalid params
        step(lambda income, cap=param(1.0, ge=0): income, name="c").run(df, params={"c": {"cap": -1.0}})
    with pytest.raises(WiringError):           # bad step name
        step(ratio, name="a/b")
    with pytest.raises(WiringError):           # two writers in a dag
        to_ir(dag(ratio, step(ratio, name="again")))
    with pytest.raises(WiringError):           # branch shape
        branch(ratio, ratio, modifies=["ratio"], name="b")


def test_shared_params_of_different_types_are_an_ir_error():
    def a(x: float, rate: float = param(1.0, shared_key="rate")) -> float:
        return x

    def b(x: float, rate: int = param(1, shared_key="rate")) -> float:
        return x

    with pytest.raises(IRError, match="shared param 'rate' is declared float by a and int by b"):
        to_ir(flow(a, b))


def test_the_package_docstring_is_a_quickstart():
    assert "flow" in decider.__doc__ and ".emit(" in decider.__doc__ and "session" in decider.__doc__


def test_engine_registry_and_expression_failures_are_caught_by_decider_error():
    with pytest.raises(EngineError, match="unknown mode 'jit'"):
        Engine().bind(flow(ratio), mode="jit")
    with pytest.raises(EngineError, match="params_validation"):
        Engine(params_validation="sometimes")
    with pytest.raises(EngineError, match="'ratio' is produced by this pipeline"):
        flow(ratio).run(pl.DataFrame({"income": [1.0], "debt": [1.0], "ratio": [0.0]}))
    with pytest.raises(RegistryError, match="not a registered"):
        ConfigurableStep.resolve("no_such_step_type")
    with pytest.raises(ExprError, match="not admitted"):
        parse("__import__('os')")
