"""Scratch tests for graph/pipeline.py: sequencing, the waterfall/version
chain (doc 03 §3.2, §3.3), the additive output frame's inputs/terminals
(§7), and `.emit()`/`.drop()`.
"""
from __future__ import annotations

import pytest

from decider2 import module
from decider2.graph.pipeline import flow
from decider2.graph.step import step


def _term_cap_step(fn_name: str, delta: float):
    def fn(term_cap: float) -> float:
        return term_cap - delta

    fn.__name__ = fn_name
    return step(output="term_cap")(fn)


# --- §3.2/§3.3: overwrite across modules is normal, and versions ------------


def test_waterfall_overwrite_across_modules_is_not_an_error():
    seed = module(_term_cap_step("seed", 0), name="seed_term_cap")
    income = module(_term_cap_step("income", 1), name="income_cap")
    sector = module(_term_cap_step("sector", 2), name="sector_cap")

    pipeline = seed | income | sector
    assert pipeline.interface.outputs == ("term_cap",)
    assert pipeline.versions()["term_cap"] == ("seed_term_cap", "income_cap", "sector_cap")


def test_final_version_is_the_default_terminal():
    seed = module(_term_cap_step("seed", 0), name="seed_term_cap")
    income = module(_term_cap_step("income", 1), name="income_cap")
    pipeline = seed | income
    # only the FINAL version is terminal — the intermediate one was consumed
    assert pipeline.interface.terminals == ("term_cap",)


# --- §2.1's "one error that matters" ----------------------------------------


def test_near_miss_across_module_boundary_is_a_build_error():
    def disposable_income(net_income: float, expenses: float) -> float:
        return net_income - expenses

    def ratio(disposible_income: float, instalment: float) -> float:  # typo
        return disposible_income / instalment

    with pytest.raises(ValueError, match="Did you mean 'disposable_income'"):
        flow(disposable_income, ratio)


# --- §7: additive output frame ----------------------------------------------


def test_consumed_intermediate_is_excluded_a_terminal_is_included():
    def a(x: float) -> float:
        return x + 1

    def b(a: float) -> float:  # consumes a's output
        return a * 2

    def c(x: float) -> float:  # independent terminal, nothing reads it
        return x - 1

    pipeline = flow(a, b, c)
    assert set(pipeline.interface.terminals) == {"b", "c"}
    assert "a" not in pipeline.interface.terminals


def test_emit_makes_a_consumed_intermediate_addressable():
    def a(x: float) -> float:
        return x + 1

    def b(a: float) -> float:
        return a * 2

    pipeline = flow(a, b)
    emitted = pipeline.emit("a")
    assert emitted.emits[0].name == "a"
    # emit() does not mutate
    assert pipeline.emits == ()


def test_emit_of_an_unproducible_name_is_a_build_error():
    def a(x: float) -> float:
        return x

    pipeline = flow(a)
    with pytest.raises(ValueError, match="no step produces"):
        pipeline.emit("does_not_exist")


def test_emit_qualified_by_a_producer_that_never_produced_it_is_an_error():
    seed = module(_term_cap_step("seed", 0), name="seed_term_cap")
    income = module(_term_cap_step("income", 1), name="income_cap")
    pipeline = seed | income
    with pytest.raises(ValueError, match="never produced by module 'nope'"):
        pipeline.emit("term_cap@nope")


def test_drop_of_a_leaf_input_is_allowed():
    def f(a: float, b: float) -> float:
        return a + b

    pipeline = flow(f)
    dropped = pipeline.drop("b")
    assert dropped.dropped == ("b",)
    assert pipeline.dropped == ()  # not mutated


def test_drop_of_an_internal_wire_is_a_build_error():
    def a(x: float) -> float:
        return x + 1

    def b(a: float) -> float:
        return a * 2

    pipeline = flow(a, b)
    with pytest.raises(ValueError, match="internal value"):
        pipeline.drop("a")


# --- flow()/compose()/`|` are the same construction path --------------------


def test_flow_and_pipe_agree():
    def a(x: float) -> float:
        return x

    def b(a: float) -> float:
        return a

    ma, mb = module(a), module(b)
    via_flow = flow(ma, mb)
    via_pipe = ma | mb
    assert via_flow.elements == via_pipe.elements


def test_flow_flattens_nested_pipelines():
    def a(x: float) -> float:
        return x

    def b(a: float) -> float:
        return a

    def c(b: float) -> float:
        return b

    inner = flow(a, b)
    outer = flow(inner, c)
    assert [m.name for m in outer.elements] == ["a", "b", "c"]


def test_pipe_mixes_module_and_bare_function():
    def a(x: float) -> float:
        return x

    def b(a: float) -> float:
        return a

    ma = module(a)
    pipeline = ma | b  # Module.__or__ -> compose
    assert [m.name for m in pipeline.elements] == ["a", "b"]
