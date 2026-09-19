"""Scratch tests for decider2.compile.codegen — doc 05 §4.2 (determinism),
§4.3 (driver shape).
"""
from __future__ import annotations

import ast

from decider2.compile import codegen
from decider2.compile.codegen import ArgRole, KernelPlan
from decider2.types import Input, NullPolicy, ParamDecl, Step


# --- module-level step functions, so `fn.__module__`/`fn.__name__` resolve
# the way a real pipeline's steps would (codegen imports by module+name).


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float, income_threshold: float) -> float:
    return min(term_cap, cap) if min_net_salary < income_threshold else term_cap


def optional_step(x: "float | None") -> float:
    return 0.0 if x is None else x


def _step(fn, *, inputs=(), params=(), reads_params=False, reads_shared=False) -> Step:
    return Step(
        name=fn.__name__,
        fn=fn,
        inputs=inputs,
        params=params,
        reads_params=reads_params,
        reads_shared=reads_shared,
    )


def test_safe_ident_handles_qualified_emit_style_names():
    assert codegen.safe_ident("term_cap@sector_cap").isidentifier()
    assert codegen.safe_ident("1abc").isidentifier()
    assert codegen.safe_ident("class").isidentifier()


def test_stable_topological_order_respects_dependencies():
    s_di = _step(disposable_income, inputs=(
        Input("net_income", float), Input("expenses", float),
    ))
    s_ar = _step(affordability_ratio, inputs=(
        Input("disposable_income", float), Input("instalment", float),
    ))
    # deliberately passed out of dependency order
    ordered = codegen.stable_topological_order([s_ar, s_di])
    assert [s.name for s in ordered] == ["disposable_income", "affordability_ratio"]


def test_stable_topological_order_is_a_pure_function_of_its_input():
    s_di = _step(disposable_income, inputs=(
        Input("net_income", float), Input("expenses", float),
    ))
    s_ar = _step(affordability_ratio, inputs=(
        Input("disposable_income", float), Input("instalment", float),
    ))
    first = [s.name for s in codegen.stable_topological_order([s_ar, s_di])]
    second = [s.name for s in codegen.stable_topological_order([s_ar, s_di])]
    assert first == second


def test_stable_topological_order_detects_a_cycle():
    a = Step(name="a", fn=disposable_income, inputs=(Input("b", float),), params=())
    b = Step(name="b", fn=affordability_ratio, inputs=(Input("a", float),), params=())
    try:
        codegen.stable_topological_order([a, b])
    except ValueError as exc:
        assert "cyclic" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a cyclic dependency")


def _flagship_plan() -> KernelPlan:
    step = _step(
        cap_by_income_band,
        inputs=(Input("term_cap", float), Input("min_net_salary", float)),
        params=(
            ParamDecl("cap", float, 48.0, None),
            ParamDecl("income_threshold", float, 5000.0, None),
        ),
    )
    return KernelPlan(
        group_name="cap_by_income_band",
        steps=(step,),
        external_inputs=step.inputs,
        required_outputs=("cap_by_income_band",),
    )


def test_emit_kernel_source_is_deterministic_byte_identical():
    plan = _flagship_plan()
    a = codegen.emit_kernel_source(plan)
    b = codegen.emit_kernel_source(plan)
    assert a == b


def test_emit_kernel_source_is_valid_python():
    source = codegen.emit_kernel_source(_flagship_plan())
    ast.parse(source)  # raises SyntaxError if this module ever emits garbage


def test_emit_kernel_source_never_uses_exec_or_spec_from_file_location():
    source = codegen.emit_kernel_source(_flagship_plan())
    assert "exec(" not in source
    assert "spec_from_file_location" not in source


def test_emit_kernel_source_never_bakes_a_param_value_into_source():
    """doc 05 §4.2: "No decision-relevant constant is emitted into driver
    source." 48.0/5000.0 are the params' *defaults*, not their compiled
    values — retuning changes an argument, not this text — so neither
    number should appear as a literal anywhere in the generated file."""
    source = codegen.emit_kernel_source(_flagship_plan())
    assert "48.0" not in source
    assert "5000.0" not in source
    assert "p_cap_by_income_band_cap" in source
    assert "p_cap_by_income_band_income_threshold" in source


def test_kernel_signature_orders_arrays_then_params_then_outputs():
    plan = _flagship_plan()
    roles = codegen.kernel_signature(plan)
    kinds = [r.kind for r in roles]
    assert kinds == ["array", "array", "param_scalar", "param_scalar", "output"]


def test_kernel_signature_adds_a_validity_array_for_optional_inputs():
    step = _step(optional_step, inputs=(Input("x", float, null_policy=NullPolicy.OPTIONAL),))
    plan = KernelPlan(
        group_name="optional_step",
        steps=(step,),
        external_inputs=step.inputs,
        required_outputs=("optional_step",),
    )
    roles = codegen.kernel_signature(plan)
    assert [r.kind for r in roles] == ["array", "valid", "output"]
    source = codegen.emit_kernel_source(plan)
    assert "if valid_x[i] else None" in source


def test_emit_kernel_source_rejects_a_group_with_no_external_inputs():
    plan = KernelPlan(group_name="empty", steps=(), external_inputs=(), required_outputs=())
    try:
        codegen.emit_kernel_source(plan)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for a kernel with no external inputs")


def test_two_steps_sharing_a_param_name_do_not_collide():
    """Params are namespaced per module instance (doc 03 §4.1) — two steps
    both declaring a `cap` param must not share one generated argument."""
    def rule_a(x: float, cap: float) -> float:
        return min(x, cap)

    def rule_b(y: float, cap: float) -> float:
        return min(y, cap)

    s_a = _step(rule_a, inputs=(Input("x", float),), params=(ParamDecl("cap", float, 1.0, None),))
    s_b = _step(rule_b, inputs=(Input("y", float),), params=(ParamDecl("cap", float, 2.0, None),))
    plan = KernelPlan(
        group_name="g",
        steps=(s_a, s_b),
        external_inputs=(Input("x", float), Input("y", float)),
        required_outputs=("rule_a", "rule_b"),
    )
    roles = codegen.kernel_signature(plan)
    param_arg_names = [r.text for r in roles if r.kind == "param_scalar"]
    assert len(param_arg_names) == len(set(param_arg_names)) == 2
