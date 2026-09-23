from __future__ import annotations

import typing as t

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from decider import engine, step
from decider.engine.ir.decls import Input, Output, ParamDecl
from decider.engine.ir.nodes import CallNode
from decider.registry import import_path
from decider.steps import ConfigurableStep, ParamRef, SequentialStep, StepRef, Value


class Threshold(ConfigurableStep):
    column: str
    cut: Value[float] = 0.5

    def to_ir(self, ctx):
        cut = ctx.value(self.cut, float)
        params, consts = ((cut,), ()) if isinstance(cut, ParamDecl) else ((), (("cut", cut),))
        return CallNode(
            ctx.origin(self), "row", _above, (Input(self.column, float),), (Output(self.name, bool),), params,
            consts=consts,
        )


class Aliased(ConfigurableStep):
    type: t.Literal["aliased_threshold"] = "aliased_threshold"

    def to_ir(self, ctx):
        return CallNode(ctx.origin(self), "row", _above, (), (Output(self.name, bool),), ())


def _above(row, params, consts):
    return (row[0] > (params or consts)[0],)


def ratio(x: float) -> float:
    return x


def test_round_trips_through_json():
    cfg = Threshold(name="hi", column="ratio", cut=ParamRef(param="hi_cut", default=0.7))
    text = cfg.model_dump_json()
    assert Threshold.model_validate_json(text).model_dump() == cfg.model_dump()
    via_ref = TypeAdapter(StepRef).validate_json(text)
    assert type(via_ref) is Threshold and via_ref.cut == ParamRef(param="hi_cut", default=0.7)


def test_is_frozen():
    cfg = Threshold(name="hi", column="ratio")
    with pytest.raises(ValidationError):
        cfg.cut = 0.9


def test_can_not_be_instantiated_without_to_ir():
    class NoIR(ConfigurableStep):
        pass

    with pytest.raises(TypeError, match="to_ir"):
        NoIR(name="x")


def test_composes_with_pipe():
    cfg = Threshold(name="hi", column="ratio")
    p = step(ratio) | cfg
    assert type(p) is SequentialStep and p.steps[1] is cfg
    node = engine.to_ir(cfg | ratio)
    assert [c.origin.path for c in node.children()] == ["hi", "ratio"]
    assert node.children()[0].origin.source == import_path(Threshold)


def test_named_and_relabel_copy_without_dumping_relabels():
    cfg = Threshold(name="hi", column="ratio")
    moved = cfg.named("lo").relabel(reads={"ratio": "dti"})
    assert (moved.name, cfg.name) == ("lo", "hi")
    assert engine.to_ir(moved).inputs[0].name == "dti"
    assert "reads" not in moved.model_dump() and "reads" not in Threshold.model_json_schema()["properties"]


def test_a_class_with_no_alias_loads_by_import_path():
    assert ConfigurableStep.resolve(import_path(Threshold)) is Threshold
    cfg = TypeAdapter(StepRef).validate_python({"type": import_path(Threshold), "name": "x", "column": "c"})
    assert type(cfg) is Threshold and cfg.model_dump()["type"] == import_path(Threshold)


def test_a_class_with_an_alias_loads_by_either_and_dumps_the_alias():
    assert ConfigurableStep.resolve("aliased_threshold") is Aliased
    assert ConfigurableStep.resolve(import_path(Aliased)) is Aliased
    cfg = TypeAdapter(StepRef).validate_python({"type": import_path(Aliased), "name": "a"})
    assert cfg.model_dump()["type"] == "aliased_threshold"


class NotAStep(BaseModel):
    type: str = "x"


@pytest.mark.parametrize("tag", ["os:system", f"{__name__}:NotAStep", f"{__name__}:ratio"])
def test_an_import_path_to_anything_else_is_rejected(tag):
    with pytest.raises(LookupError, match="not a registered"):
        ConfigurableStep.resolve(tag)


def test_a_duplicate_alias_raises():
    with pytest.raises(ValueError, match="aliased_threshold"):
        class Again(ConfigurableStep):
            type: t.Literal["aliased_threshold"] = "aliased_threshold"

            def to_ir(self, ctx):
                return None
