"""
Integration tests for parameters and computed features.

Covers:
- InputRef resolving from a default parameter value
- InputRef resolving from a runtime struct column
- Runtime struct overriding default
- Computed feature expressions (two-column arithmetic, parameter access)
"""

import polars as pl
import pytest
from decider.modules.rules import (
    FlatRuleModule,
    UnaryRule,
    LeafRule,
    RuleRoot,
    RuleMeta,
    TreeOutput,
    UnaryLessThan,
    UnaryBetween,
    InputRef,
    ParameterInfo,
    Feature,
)
from decider.serializable.schema import PrimitiveSchema


def _output(*labels: str, default: str = "default") -> TreeOutput:
    return TreeOutput(
        data=[{"r": lbl} for lbl in labels],
        default={"r": default},
        dtypes=[("r", "String")],
        type_defs={},
    )


def _module(rule, output, parameters=None):
    return FlatRuleModule(
        output=output,
        rule=RuleRoot(meta=RuleMeta(), rule=rule),
        parameters=parameters or {},
    )


# ---------------------------------------------------------------------------
# InputRef — parameter-based thresholds
# ---------------------------------------------------------------------------

def test_inputref_uses_default_when_no_runtime_column():
    """When no parameters column in the frame, the default value is used."""
    rule = UnaryRule(
        condition=UnaryLessThan(feature="score", threshold=InputRef(key="thresh")),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1),
    )
    m = _module(
        rule,
        _output("low", "high"),
        parameters={"thresh": ParameterInfo(type=PrimitiveSchema(type="Float64"), default_value=50.0)},
    )
    df = pl.DataFrame({"score": [30.0, 70.0]})
    result = m({"input": df.lazy()})["r"].to_list()
    assert result == ["low", "high"]


def test_inputref_uses_runtime_struct_column():
    """When a parameters struct column is present, it overrides the default."""
    rule = UnaryRule(
        condition=UnaryLessThan(feature="score", threshold=InputRef(key="thresh")),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1),
    )
    m = _module(
        rule,
        _output("low", "high"),
        parameters={"thresh": ParameterInfo(type=PrimitiveSchema(type="Float64"), default_value=50.0)},
    )
    # Pass thresh=80 at runtime — both rows should now be "low"
    df = pl.DataFrame({
        "score": [30.0, 70.0],
        "parameters": [{"thresh": 80.0}, {"thresh": 80.0}],
    })
    result = m({"input": df.lazy()})["r"].to_list()
    assert result == ["low", "low"]


def test_inputref_runtime_overrides_default_per_row():
    """Runtime parameter can differ per row, overriding the default independently."""
    rule = UnaryRule(
        condition=UnaryLessThan(feature="score", threshold=InputRef(key="thresh")),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1),
    )
    m = _module(
        rule,
        _output("low", "high"),
        parameters={"thresh": ParameterInfo(type=PrimitiveSchema(type="Float64"), default_value=50.0)},
    )
    # Row 0: thresh=20 → score=30 is NOT < 20 → "high"
    # Row 1: thresh=80 → score=70 IS < 80 → "low"
    df = pl.DataFrame({
        "score": [30.0, 70.0],
        "parameters": [{"thresh": 20.0}, {"thresh": 80.0}],
    })
    result = m({"input": df.lazy()})["r"].to_list()
    assert result == ["high", "low"]


def test_inputref_between_with_two_parameters():
    """Both min and max of a Between condition can be InputRef-bound."""
    rule = UnaryRule(
        condition=UnaryBetween(feature="score", min=InputRef(key="lo"), max=InputRef(key="hi")),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=-1),
    )
    m = _module(
        rule,
        _output("in_range", default="out"),
        parameters={
            "lo": ParameterInfo(type=PrimitiveSchema(type="Float64"), default_value=20.0),
            "hi": ParameterInfo(type=PrimitiveSchema(type="Float64"), default_value=80.0),
        },
    )
    df = pl.DataFrame({"score": [10.0, 50.0, 90.0]})
    result = m({"input": df.lazy()})["r"].to_list()
    assert result == ["out", "in_range", "out"]


try:
    import simpleeval as _simpleeval  # noqa: F401
    _SIMPLEEVAL = True
except ImportError:
    _SIMPLEEVAL = False

_skip_simpleeval = pytest.mark.skipif(not _SIMPLEEVAL, reason="simpleeval not installed")


# ---------------------------------------------------------------------------
# Computed features
# ---------------------------------------------------------------------------

@_skip_simpleeval
def test_computed_feature_two_column_expression():
    """A computed feature that combines two columns is evaluated correctly."""
    # feature: "amount * quantity" > 100 → "large"
    rule = UnaryRule(
        condition=UnaryLessThan(
            feature=Feature.model_validate({"type": "computed", "expression": "amount * quantity"}),
            threshold=100.0,
        ),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1),
    )
    m = _module(rule, _output("small", "large"))
    df = pl.DataFrame({"amount": [5.0, 20.0], "quantity": [10.0, 10.0]})
    # 5*10=50 < 100 → small; 20*10=200 >= 100 → large
    result = m({"input": df.lazy()})["r"].to_list()
    assert result == ["small", "large"]


@_skip_simpleeval
def test_computed_feature_uses_parameter():
    """A computed feature referencing p.key uses the parameter value correctly."""
    # feature: "amount + p.bonus"
    rule = UnaryRule(
        condition=UnaryLessThan(
            feature=Feature.model_validate({"type": "computed", "expression": "amount + p.bonus"}),
            threshold=100.0,
        ),
        then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=1),
    )
    m = _module(
        rule,
        _output("small", "large"),
        parameters={"bonus": ParameterInfo(type=PrimitiveSchema(type="Float64"), default_value=10.0)},
    )
    # amount=80, bonus=10 → 90 < 100 → small
    # amount=95, bonus=10 → 105 >= 100 → large
    df = pl.DataFrame({"amount": [80.0, 95.0]})
    result = m({"input": df.lazy()})["r"].to_list()
    assert result == ["small", "large"]
