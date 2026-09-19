"""Scratch tests for graph/module.py, graph/step.py, graph/interface.py.

Not the acceptance test (tests/test_flagship.py) — these exercise the
graph/ module in isolation, including edge cases the flagship pipeline
doesn't touch: duplicate outputs (doc 03 §3.1), the unbound-input
did-you-mean (§2.2), name derivation rules (§5), `.bind()`/`.relabel()`
(§4.3/§5.2), and `contract=` (§5.1).
"""
from __future__ import annotations

import json

import pytest

from decider2 import missing_as, module, not_applicable_as, param
from decider2.graph.pipeline import flow
from decider2.graph.step import step
from decider2.types import NullPolicy


# --- name derivation (doc 03 §5) -------------------------------------------


def test_single_step_module_derives_its_name_from_the_step():
    def f(a: float) -> float:
        return a

    m = module(f)
    assert m.name == "f"


def test_multi_step_module_requires_name():
    def f(a: float) -> float:
        return a

    def g(b: float) -> float:
        return b

    with pytest.raises(ValueError, match="needs name="):
        module(f, g)


def test_restating_the_derived_name_is_a_lint_error():
    def f(a: float) -> float:
        return a

    with pytest.raises(ValueError, match="restates the derived name"):
        module(f, name="f")


def test_step_output_can_be_overridden():
    @step(output="term_cap")
    def apply_income_cap(term_cap: float) -> float:
        return term_cap - 1

    m = module(apply_income_cap, name="income_cap")
    assert m.interface.outputs == ("term_cap",)
    assert m.name == "income_cap"


# --- §3.1: no overwrite inside a module -------------------------------------


def test_duplicate_output_inside_one_module_is_a_build_error():
    @step(output="x")
    def a(y: float) -> float:
        return y

    def x(z: float) -> float:
        return z

    with pytest.raises(ValueError, match="declare output 'x'"):
        module(x, a, name="dup")


# --- §2.2: the did-you-mean ---------------------------------------------


def test_near_miss_unbound_input_is_a_build_error_with_a_suggestion():
    def disposable_income(net_income: float, expenses: float) -> float:
        return net_income - expenses

    def ratio(disposible_income: float, instalment: float) -> float:  # typo
        return disposible_income / instalment

    with pytest.raises(ValueError, match="Did you mean 'disposable_income'"):
        module(disposable_income, ratio, name="m")


def test_genuinely_unrelated_unbound_input_is_not_an_error():
    def disposable_income(net_income: float, expenses: float) -> float:
        return net_income - expenses

    def ratio(disposable_income: float, instalment: float) -> float:
        return disposable_income / instalment

    m = module(disposable_income, ratio, name="m")
    assert "instalment" in [i.name for i in m.interface.inputs]


# --- null-policy tiers flow through (doc 03 §1) -----------------------------


def test_all_four_null_tiers_are_classified():
    def tiers(
        a: float,
        b: float = missing_as(0.0),
        c: float | None = None,
        d: float = not_applicable_as(1.0),
    ) -> float:
        return a

    m = module(tiers)
    by_name = {i.name: i for i in m.interface.inputs}
    assert by_name["a"].null_policy is NullPolicy.REQUIRED
    assert by_name["b"].null_policy is NullPolicy.MISSING_AS and by_name["b"].fill == 0.0
    assert by_name["c"].null_policy is NullPolicy.OPTIONAL
    assert by_name["d"].null_policy is NullPolicy.NOT_APPLICABLE_AS and by_name["d"].fill == 1.0


# --- §4.3 .bind(), §5.2 .relabel() -------------------------------------


def test_bind_removes_a_param_from_the_caller_facing_schema():
    def f(a: float, k: float = param(2.0, ge=0)) -> float:
        return a * k

    m = module(f)
    assert m.params_schema() == {"k": 2.0}
    bound = m.bind(k=5.0)
    assert bound.params_schema() == {}
    assert m.params_schema() == {"k": 2.0}  # original untouched


def test_bind_rejects_an_unknown_param():
    def f(a: float, k: float = param(2.0)) -> float:
        return a * k

    m = module(f)
    with pytest.raises(ValueError, match="unknown param"):
        m.bind(not_a_param=1.0)


def test_relabel_reads_renames_an_input_without_touching_the_original():
    def f(net_income: float) -> float:
        return net_income

    m = module(f)
    relabelled = m.relabel(reads={"net_income": "monthly_net_salary"})
    assert [i.name for i in relabelled.interface.inputs] == ["monthly_net_salary"]
    assert [i.name for i in m.interface.inputs] == ["net_income"]


def test_relabel_writes_renames_an_output():
    def f(a: float) -> float:
        return a

    m = module(f).relabel(writes={"f": "renamed_output"})
    assert m.interface.outputs == ("renamed_output",)


# --- reuse: __call__ renames an instance (doc 03 §5.3) ----------------------


def test_calling_a_module_renames_it_for_reuse():
    def f(a: float) -> float:
        return a

    m = module(f)
    m2 = m(name="f_second")
    assert m2.name == "f_second"
    assert m.name == "f"
    assert m2.steps == m.steps  # same interior, per §5.2's "interior untouched"


def test_reusing_the_same_module_instance_twice_in_a_pipeline_is_an_error():
    def f(a: float) -> float:
        return a

    m = module(f)
    with pytest.raises(ValueError, match="twice"):
        flow(m, m)


def test_renamed_reuse_is_allowed():
    def f(a: float) -> float:
        return a

    m = module(f)
    p = flow(m, m(name="f_second"))
    assert {mod.name for mod in p.elements} == {"f", "f_second"}


# --- contract= (doc 03 §5.1) ------------------------------------------------


def test_contract_true_snapshots_to_the_derived_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def f(a: float, b: float) -> float:
        return a + b

    module(f, contract=True)
    snapshot_path = tmp_path / "contracts" / "f.json"
    assert snapshot_path.exists()
    data = json.loads(snapshot_path.read_text())
    assert data["outputs"] == ["f"]


def test_contract_catches_a_breaking_change(tmp_path):
    path = tmp_path / "contract.json"

    def f(a: float, b: float) -> float:
        return a + b

    module(f, contract=str(path))

    def f2(a: float, b: float, c: float) -> float:
        return a + b + c

    f2.__name__ = "f"
    with pytest.raises(ValueError, match="no longer matches"):
        module(f2, contract=str(path))
