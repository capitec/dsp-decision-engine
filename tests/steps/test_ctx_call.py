"""`IRContext.call`: a config step built from one function using only public imports."""
from __future__ import annotations

from typing import Literal

import polars as pl
import pytest

from decider import ConfigurableStep, Value, missing_as
from decider.engine import to_ir
from decider.engine.ir import CallNode, NullPolicy, ParamDecl
from decider.exceptions import IRError


def scale(x: float, factor: float) -> float:
    return x * factor


class Scaled(ConfigurableStep):
    type: Literal["scaled"] = "scaled"
    column: str
    factor: Value[float] = 1.0

    def to_ir(self, ctx):
        return ctx.call(self, scale, inputs={"x": self.column}, values={"factor": self.factor})


DF = pl.DataFrame({"income": [100.0, 250.0]})


def test_the_documented_example_runs_retunes_and_round_trips():
    cfg = ConfigurableStep.load({"type": "scaled", "name": "doubled", "column": "income",
                                 "factor": {"param": "factor", "default": 2.0}})
    assert cfg.run(DF)["doubled"].to_list() == [200.0, 500.0]
    assert cfg.run(DF, params={"doubled": {"factor": 3.0}})["doubled"].to_list() == [300.0, 750.0]
    assert ConfigurableStep.load(cfg.model_dump_json()) == cfg
    assert "reads=" not in repr(cfg) and "writes=" not in repr(cfg)


def test_a_literal_is_a_const_and_a_ref_is_a_param_fed_to_its_argument():
    literal = to_ir(Scaled(name="s", column="income", factor=2.0))
    assert isinstance(literal, CallNode) and literal.kind == "scalar"
    assert literal.consts == (("factor", 2.0),) and literal.params == ()
    assert [(i.name, i.arg) for i in literal.inputs] == [("income", "x")]
    ref = to_ir(Scaled(name="s", column="income", factor={"param": "k", "default": 2.0}))
    assert ref.params == (ParamDecl("k", float, 2.0, arg="factor"),) and ref.consts == ()


def test_a_required_ref_takes_its_type_from_the_function_signature():
    node = to_ir(Scaled(name="s", column="income", factor={"param": "k"}))
    assert node.params[0].annotation is float and node.params[0].required


def test_null_policies_and_multiple_outputs_come_from_the_signature():
    def split(cut: float, x: float = missing_as(0.0)) -> tuple[bool, float]:
        return x > cut, x - cut

    class Split(ConfigurableStep):
        cut: Value[float]

        def to_ir(self, ctx):
            return ctx.call(self, split, outputs=["over", "margin"], values={"cut": self.cut})

    node = to_ir(Split(name="s", cut=1.0))
    assert node.inputs[0].null_policy is NullPolicy.MISSING_AS
    assert [(o.name, o.annotation) for o in node.outputs] == [("over", bool), ("margin", float)]


def test_wiring_an_argument_the_function_lacks_is_an_error():
    class Bad(ConfigurableStep):
        def to_ir(self, ctx):
            return ctx.call(self, scale, values={"factr": 2.0})

    with pytest.raises(IRError, match=r"bad: scale has no argument\(s\) \['factr'\]; its inputs are \['factor', 'x'\]"):
        to_ir(Bad(name="bad"))
