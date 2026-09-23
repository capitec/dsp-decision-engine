"""`emit` and `drop` resolved to versions and output columns."""
from __future__ import annotations

import pytest

from decider import flow, param, step
from decider.engine.wiring import resolve


def a(x: float) -> float:
    return x + 1


def b(a: float) -> float:
    return a * 2


@step(output="term_cap")
def product_ceiling(requested_term: float, ceiling: float = param(60.0)) -> float:
    return min(requested_term, ceiling)


@step(output="term_cap")
def cap_by_income(term_cap: float, min_net_salary: float, cap: float = param(48.0)) -> float:
    return min(term_cap, cap) if min_net_salary < 5000 else term_cap


def test_emit_makes_a_consumed_intermediate_an_output():
    plan = resolve(flow(a, b).emit("a"))
    assert plan.outputs["a"] is plan.calls[0].writes[0]
    assert "a" not in resolve(flow(a, b)).outputs


def test_emit_every_version_names_each_column_by_its_producer():
    plan = resolve(flow(product_ceiling, cap_by_income).emit("term_cap@*"))
    first, second = plan.chains["term_cap"]
    assert plan.outputs["term_cap@product_ceiling"] is first
    assert plan.outputs["term_cap@cap_by_income"] is second
    assert plan.outputs["term_cap"] is second


def test_emit_one_version_by_its_producer():
    plan = resolve(flow(product_ceiling, cap_by_income).emit("term_cap@product_ceiling"))
    assert plan.outputs["term_cap@product_ceiling"] is plan.chains["term_cap"][0]
    assert "term_cap@cap_by_income" not in plan.outputs


def test_a_qualifier_is_relative_to_the_flow_that_emits():
    inner = flow(product_ceiling, cap_by_income, name="term").emit("term_cap@product_ceiling")
    plan = resolve(flow(inner, name="outer"))
    assert plan.outputs["term_cap@product_ceiling"].producer == "outer/term/product_ceiling"


def test_a_qualifier_on_the_root_is_the_full_path():
    inner = flow(product_ceiling, cap_by_income, name="term")
    plan = resolve((step(a) | inner).emit("term_cap@*"))
    assert {k for k in plan.outputs if "@" in k} == {"term_cap@term/product_ceiling", "term_cap@term/cap_by_income"}


def test_emit_of_an_unknown_name_is_an_error_with_a_suggestion():
    with pytest.raises(ValueError, match=r"^<root>: emit\('ab'\): no step produces 'ab'.*Did you mean 'b'"):
        resolve(flow(a, b).emit("ab"))


def test_emit_qualified_by_a_step_that_never_wrote_it_lists_the_producers():
    with pytest.raises(ValueError, match=r"never produced by 'cap_by_incme'.*Did you mean 'cap_by_income'"):
        resolve(flow(product_ceiling, cap_by_income).emit("term_cap@cap_by_incme"))


def test_emit_qualified_for_an_input_column_is_an_error():
    with pytest.raises(ValueError, match="no step produces 'x'"):
        resolve(flow(a, b).emit("x@a"))


def test_emit_of_an_input_column_is_allowed():
    plan = resolve(flow(a, b).emit("x"))
    assert plan.outputs["x"].producer is None


def test_drop_of_an_input_column_removes_it():
    def f(a: float, b: float) -> float:
        return a + b

    plan = resolve(flow(f).drop("b"))
    assert "b" not in plan.outputs
    assert plan.drops == ("b",)


def test_drop_of_an_output_removes_it():
    plan = resolve(flow(a, b).drop("b"))
    assert set(plan.outputs) == {"x"}


def test_drop_of_an_internal_value_is_an_error():
    with pytest.raises(ValueError, match=r"^<root>: drop\('a'\): 'a' is an internal value"):
        resolve(flow(a, b).drop("a"))


def test_drop_of_an_emitted_intermediate_is_allowed():
    plan = resolve(flow(a, b).emit("a").drop("a"))
    assert "a" not in plan.outputs


def test_a_misspelled_drop_is_an_error_with_a_suggestion():
    def affordable(x: float) -> bool:
        return x > 0

    with pytest.raises(ValueError, match=r"drop\('afordable'\): no such value. Did you mean 'affordable'"):
        resolve(flow(affordable, a).drop("afordable"))


def test_drop_of_an_unrelated_name_is_kept_for_frame_columns():
    plan = resolve(flow(a, b).drop("client_notes"))
    assert plan.drops == ("client_notes",)


def test_emit_accepts_the_absolute_path_that_step_map_and_reports_use():
    term = flow(product_ceiling, cap_by_income, name="term")
    pipeline = flow(term.emit("term_cap@app/term/product_ceiling"), name="app")
    plan = resolve(pipeline)
    assert plan.outputs["term_cap@app/term/product_ceiling"] is plan.chains["term_cap"][0]
    root = resolve(flow(product_ceiling, cap_by_income, name="term").emit("term_cap@term/cap_by_income"))
    assert root.outputs["term_cap@term/cap_by_income"] is root.chains["term_cap"][1]


def test_emit_of_an_unknown_path_suggests_the_nearest_relative_or_absolute_one():
    term = flow(product_ceiling, cap_by_income, name="term")
    with pytest.raises(ValueError, match="Did you mean 'term/cap_by_income'"):
        resolve(term.emit("term_cap@term/cap_by_incme"))
    with pytest.raises(ValueError, match="Did you mean 'cap_by_income'"):
        resolve(term.emit("term_cap@cap_by_incme"))
