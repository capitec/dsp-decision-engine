"""Library errors share `DecikitError` and keep their builtin base, so either `except` catches them."""
from __future__ import annotations

import polars as pl
import pytest

import decikit
from decikit import branch, dag, flow, param, step
from decikit._engine import Engine, FusedEngine, to_ir
from decikit._engine.boundary._arrow.plan import ArrowKindError as ArrowKindFromPlan
from decikit._engine.boundary.dtypes import NeedsKernelSplit as NeedsKernelSplitFromDtypes
from decikit._engine.boundary.nulls import MissingInputError as MissingFromNulls
from decikit._engine.params import ParamsError as ParamsFromParams
from decikit._engine.boundary.errors import ArrowImportError, ArrowKindError, NeedsKernelSplit
from decikit.exceptions import (
    DecikitError, EngineError, ExprError, IRError, MissingInputError, ParamsError, RegistryError, WiringError,
)
from decikit.serving._errors import (
    InputParsingError, OutputFormattingError, UnsupportedAcceptError, UnsupportedContentTypeError,
)
from decikit.steps import ConfigurableStep
from decikit.steps.expr import ExprError as ExprFromExpr, parse


@pytest.mark.parametrize("error, builtin", [
    (WiringError, ValueError), (IRError, TypeError), (ParamsError, ValueError), (MissingInputError, ValueError),
    (ArrowKindError, TypeError), (NeedsKernelSplit, TypeError), (ArrowImportError, RuntimeError),
    (EngineError, ValueError), (RegistryError, LookupError), (ExprError, ValueError),
    (UnsupportedContentTypeError, Exception), (InputParsingError, Exception), (UnsupportedAcceptError, Exception),
    (OutputFormattingError, Exception),
])
def test_each_library_error_is_a_decikit_error_and_its_builtin(error, builtin):
    assert issubclass(error, DecikitError) and issubclass(error, builtin)


def test_the_old_import_locations_name_the_same_classes():
    assert ArrowKindFromPlan is ArrowKindError and NeedsKernelSplitFromDtypes is NeedsKernelSplit
    assert MissingFromNulls is MissingInputError and ParamsFromParams is ParamsError
    assert ExprFromExpr is ExprError


def ratio(income: float, debt: float) -> float:
    return debt / income


def test_build_and_run_failures_are_caught_by_decikit_error():
    df = pl.DataFrame({"income": [1.0]})
    with pytest.raises(DecikitError):          # missing input column
        flow(ratio).run(df)
    with pytest.raises(DecikitError):          # invalid params
        step(lambda income, cap=param(1.0, ge=0): income, name="c").run(df, params={"c": {"cap": -1.0}})
    with pytest.raises(WiringError):           # bad step name
        step(ratio, name="a/b")
    with pytest.raises(WiringError):           # two writers in a dag
        to_ir(dag(ratio, step(ratio, name="again")))
    with pytest.raises(WiringError):           # branch shape
        branch(ratio, modifies=["ratio"], name="b")


def test_shared_params_of_different_types_are_an_ir_error():
    def a(x: float, rate: float = param(1.0, shared_key="rate")) -> float:
        return x

    def b(x: float, rate: int = param(1, shared_key="rate")) -> float:
        return x

    with pytest.raises(IRError, match="shared param 'rate' is declared float by a and int by b"):
        to_ir(flow(a, b))


def test_the_package_docstring_is_a_quickstart():
    assert "flow" in decikit.__doc__ and ".emit(" in decikit.__doc__ and "session" in decikit.__doc__


def test_engine_registry_and_expression_failures_are_caught_by_decikit_error():
    with pytest.raises(RegistryError, match="Did you mean 'fused'"):
        Engine.resolve("fsued")
    with pytest.raises(ValueError, match="params_validation"):
        FusedEngine(params_validation="sometimes")
    with pytest.raises(EngineError, match="'ratio' is produced by this pipeline"):
        flow(ratio).run(pl.DataFrame({"income": [1.0], "debt": [1.0], "ratio": [0.0]}))
    with pytest.raises(RegistryError, match="not a registered"):
        ConfigurableStep.resolve("no_such_step_type")
    with pytest.raises(ExprError, match="not admitted"):
        parse("__import__('os')")
