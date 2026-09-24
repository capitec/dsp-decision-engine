"""Build-time wiring errors: typos, forward references, and the waterfalls that must stay legal."""
from __future__ import annotations

import pytest

from decider import dag, flow, param, step
from decider.engine.wiring import resolve


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def ratio(disposible_income: float, instalment: float) -> float:  # typo
    return disposible_income / instalment


def test_a_near_miss_of_an_earlier_output_is_a_typo_with_a_suggestion():
    with pytest.raises(ValueError, match="Did you mean 'disposable_income'") as e:
        resolve(flow(disposable_income, ratio))
    assert str(e.value).startswith("ratio: input 'disposible_income'")


def test_a_near_miss_inside_a_dag_names_the_node_path():
    with pytest.raises(ValueError, match=r"^m/ratio: .*Did you mean 'disposable_income'"):
        resolve(dag(disposable_income, ratio, name="m"))


def test_a_near_miss_across_a_named_flow_boundary_is_a_typo():
    with pytest.raises(ValueError, match=r"^second/ratio: .*produced by 'first/disposable_income'"):
        resolve(flow(flow(disposable_income, name="first"), flow(ratio, name="second")))


def test_a_genuinely_unrelated_unknown_name_is_an_input_column():
    def affordability(disposable_income: float, instalment: float) -> float:
        return disposable_income / instalment

    plan = resolve(dag(disposable_income, affordability, name="m"))
    assert "instalment" in [i.name for i in plan.inputs]


def initiation_fee(offered_amount: float) -> float:
    return offered_amount * 0.015


def offered_amount(requested_amount: float, cap: float = param(200000.0)) -> float:
    return min(requested_amount, cap)


def test_reading_an_input_column_a_later_step_writes_is_an_error_naming_both():
    with pytest.raises(ValueError, match="offered_amount") as e:
        resolve(flow(initiation_fee, offered_amount))
    assert "initiation_fee" in str(e.value)


def test_reading_an_input_then_narrowing_it_later_is_a_forward_reference():
    def peek(term_cap: float) -> float:
        return term_cap

    @step(output="term_cap")
    def cap(term_cap: float) -> float:
        return term_cap

    with pytest.raises(ValueError, match="peek reads 'term_cap' as an input column, but cap"):
        resolve(flow(peek, cap))


def test_a_forward_reference_from_a_branch_condition_is_caught_after_the_branch():
    from decider import branch

    def is_big(amount: float) -> bool:
        return amount > 1

    @step(output="x")
    def one(x: float) -> float:
        return x

    @step(output="x")
    def two(x: float) -> float:
        return x

    def amount(y: float) -> float:
        return y

    with pytest.raises(ValueError, match="by/is_big reads 'amount'"):
        resolve(flow(branch(is_big, one, two, modifies=["x"], name="by"), amount))


def _term_cap_step(name: str, delta: float):
    def fn(term_cap: float) -> float:
        return term_cap - delta

    fn.__name__ = name
    return step(output="term_cap")(fn)


def test_a_chain_of_self_read_waterfalls_is_no_forward_reference():
    plan = resolve(flow(_term_cap_step("seed", 0), _term_cap_step("income", 1), _term_cap_step("sector", 2)))
    assert [v.producer for v in plan.chains["term_cap"]] == ["seed", "income", "sector"]


def test_an_input_read_with_two_types_is_an_error_naming_both_readers():
    def as_float(dependants: float) -> float:
        return dependants * 1.5

    def as_int(dependants: int | None) -> str:
        return str(dependants)

    with pytest.raises(ValueError, match=r"as_float reads input column 'dependants' as float, but as_int reads it as int"):
        resolve(flow(as_float, as_int))


def test_an_input_read_with_one_type_and_its_optional_form_is_fine():
    def a(n: int) -> int:
        return n

    def b(n: int | None) -> int:
        return n or 0

    assert [i.name for i in resolve(flow(a, b)).inputs] == ["n"]


def test_a_similar_but_not_mistyped_name_is_an_input_column_with_a_warning():
    def entity_base_score() -> float:
        return 600.0

    def total(entity_base_score: float, entity_bureau_score: float) -> float:
        return entity_base_score + entity_bureau_score

    with pytest.warns(UserWarning, match="reading 'entity_bureau_score' as an input column"):
        plan = resolve(flow(entity_base_score, total))
    assert [i.name for i in plan.inputs] == ["entity_bureau_score"]


def test_names_differing_only_in_digits_are_no_typo():
    def applicant1_income(x: float) -> float:
        return x

    def total(applicant1_income: float, applicant2_income: float) -> float:
        return applicant1_income + applicant2_income

    with pytest.warns(UserWarning):
        plan = resolve(flow(applicant1_income, total))
    assert "applicant2_income" in [i.name for i in plan.inputs]
