from __future__ import annotations

import dataclasses

import pytest

from decider import dag, engine, flow, param, step
from decider.engine.ir.decls import NullPolicy
from decider.steps import FunctionStep, as_step


def cap_by_income_band(term_cap: float, min_net_salary: float,
                       cap: float = param(48.0, ge=6, le=60),
                       income_threshold: float = param(5000.0, ge=0)) -> float:
    return min(term_cap, cap) if min_net_salary < income_threshold else term_cap


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def _fields(s: FunctionStep) -> list:
    return [getattr(s, f.name) for f in dataclasses.fields(s)]


def test_every_spelling_of_step_produces_identical_fields():
    spellings = [
        step(cap_by_income_band),
        step()(cap_by_income_band),
        as_step(cap_by_income_band),
        flow(cap_by_income_band).steps[0],
        dag(cap_by_income_band),
    ]
    assert all(type(s) is FunctionStep for s in spellings)
    assert all(_fields(s) == _fields(spellings[0]) for s in spellings)
    assert spellings[0].name == "cap_by_income_band"
    assert spellings[0].outputs == ("cap_by_income_band",)


def test_a_step_is_callable_with_its_param_defaults():
    s = step(cap_by_income_band)
    assert s(term_cap=60.0, min_net_salary=4000.0) == 48.0
    assert s(term_cap=60.0, min_net_salary=4000.0, cap=36.0) == 36.0
    assert s(term_cap=60.0, min_net_salary=9200.0) == 60.0


def test_a_step_is_callable_with_bool_none_and_required_params():
    @step
    def gate(x: float, on: bool = param(True), floor: float | None = param(None),
             scale: float = param(required=True)) -> float:
        return x * scale if on and floor is None else 0.0

    assert gate(2.0, scale=3.0) == 6.0
    with pytest.raises(TypeError, match="required param 'scale'"):
        gate(2.0)


def test_output_overrides_the_output_name_but_not_the_step_name():
    @step(output="term_cap")
    def apply_income_cap(term_cap: float) -> float:
        return term_cap - 1

    assert apply_income_cap.name == "apply_income_cap"
    node = engine.to_ir(apply_income_cap)
    assert [o.name for o in node.outputs] == ["term_cap"]
    assert [i.name for i in node.inputs] == ["term_cap"]


def test_name_and_nogil_reach_the_ir():
    node = engine.to_ir(step(cap_by_income_band, name="income_band", nogil=True))
    assert node.origin.path == "income_band"
    assert node.nogil and node.kind == "scalar" and node.fn is cap_by_income_band


def test_multiple_outputs_with_a_tuple_annotation():
    @step(outputs=("a", "b"))
    def split(x: float) -> tuple[float, bool]:
        return x, x > 0

    node = engine.to_ir(split)
    assert [(o.name, o.annotation) for o in node.outputs] == [("a", float), ("b", bool)]
    assert split(x=1.0) == (1.0, True)


def test_a_mismatched_output_annotation_raises_when_the_step_is_built():
    def split(x: float) -> tuple[float]:
        return (x,)

    with pytest.raises(TypeError, match="tuple"):
        step(outputs=("a", "b"))(split)


def test_output_and_outputs_together_raise():
    with pytest.raises(TypeError, match="not both"):
        step(output="a", outputs=("a", "b"))


def test_inputs_carry_their_null_policies_into_the_ir():
    from decider import missing_as

    def tiers(a: float, b: float = missing_as(0.0), c: float | None = None) -> float:
        return a + b + (c or 0.0)

    by_name = {i.name: i for i in engine.to_ir(tiers).inputs}
    assert by_name["a"].null_policy is NullPolicy.REQUIRED
    assert by_name["b"].null_policy is NullPolicy.MISSING_AS and by_name["b"].fill == 0.0
    assert by_name["c"].null_policy is NullPolicy.OPTIONAL


def test_params_are_namespaced_by_the_step_path():
    schema = flow(affordability_ratio, cap_by_income_band).parameters()
    assert schema["cap_by_income_band"]["cap"]["default"] == 48.0
    assert schema["cap_by_income_band"]["income_threshold"]["default"] == 5000.0


def test_bind_sets_a_params_default_without_touching_the_original():
    s = step(cap_by_income_band)
    bound = s.bind(cap=36.0)
    assert bound.parameters()["cap_by_income_band"]["cap"]["default"] == 36.0
    assert s.parameters()["cap_by_income_band"]["cap"]["default"] == 48.0
    assert bound(term_cap=60.0, min_net_salary=4000.0) == 36.0
    decl = engine.to_ir(bound).params[0]
    assert decl.field_info.default == 36.0
    assert decl.field_info.metadata == engine.to_ir(s).params[0].field_info.metadata


def test_bind_rejects_an_unknown_param():
    with pytest.raises(ValueError, match="unknown param"):
        step(cap_by_income_band).bind(not_a_param=1.0)


def test_named_renames_a_copy():
    s = step(cap_by_income_band)
    again = s.named("again")
    assert (again.name, s.name) == ("again", "cap_by_income_band")
    assert again.fn is s.fn


def test_a_named_lambda_writes_its_name_while_a_named_function_keeps_its_own():
    assert step(lambda x: x * 2, name="dbl").outputs == ("dbl",)
    assert step(lambda x: x * 2, name="dbl", output="twice").outputs == ("twice",)
    assert step(affordability_ratio, name="ratio").outputs == ("affordability_ratio",)
