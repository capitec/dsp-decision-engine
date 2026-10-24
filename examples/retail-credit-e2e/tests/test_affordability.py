"""Affordability: capacity from discretionary income to the verdict, and its overlays."""
import pytest

PASS, MARGINAL, FAIL, INDETERMINATE = 1, 2, 3, 4


def test_the_sample_capacity_chain_to_the_cent(decide):
    out = decide()
    assert out["discretionary_income"] == 8213.5          # 16 013.50 net - 6 200 expenses - 1 600 obligations
    assert out["pre_buffer"] == out["ratio_ceiling"] == 3394.86  # the grade-7 ratio (19.4%) of net income binds, not capacity
    assert out["residual_floor"] == 2400.0 and out["affordability_binding_constraint_code"] == 2
    assert out["affordability_buffer_unadjusted"] == 0.2 and out["affordability_buffer_applied"] == pytest.approx(0.24)
    assert out["buffer_adjustments_applied"] == ["ADJ-0114"]
    assert out["max_affordable_instalment"] == 2580.09    # 3 394.86 x (1 - 0.24)
    assert out["affordability_verdict_code"] == PASS


def test_the_buffer_overlay_is_a_param_switch_through_the_same_code(decide, params):
    off = {"retail_credit": {**params["retail_credit"], "limits": {
        **params["retail_credit"]["limits"], "apply_affordability_buffer_adjustments": {"adjustment_stack_enabled": False}}}}
    out = decide(params_document=off)
    assert out["affordability_buffer_applied"] == 0.2 and out["buffer_adjustments_applied"] == []
    assert out["max_affordable_instalment"] == 2715.89


def test_capacity_below_the_ratio_ceiling_binds_instead(decide):
    out = decide(declared_living_expenses=10_000.0)
    assert out["pre_buffer"] == out["capacity"] == 2013.5 and out["affordability_binding_constraint_code"] == 1


def test_a_self_employed_earner_and_a_joint_household_get_a_lower_instalment_cap(decide):
    assert decide()["instalment_cap"] == decide()["max_affordable_instalment"]
    for change, factor in [({"self_employed_haircut": True}, 0.90), ({"is_joint_application": True}, 0.95),
                           ({"self_employed_haircut": True, "is_joint_application": True}, 0.90 * 0.95)]:
        out = decide(**change)
        assert out["instalment_cap"] == round(out["max_affordable_instalment"] * factor, 2)


def test_the_offer_is_solved_against_the_instalment_cap_not_the_uncapped_figure(decide):
    out = decide(self_employed_haircut=True)
    assert out["instalment"] <= out["instalment_cap"] < out["max_affordable_instalment"]
    assert out["offer_amount"] < decide()["offer_amount"] and out["outcome_code"] == 1


@pytest.mark.parametrize("change, verdict", [
    ({"payslip_income": None, "declared_income": 18_500.0, "bureau_score": 800.0}, MARGINAL),  # declared income only
    ({"payslip_income": None}, INDETERMINATE),                                                  # no income established
    ({"declared_living_expenses": 15_900.0}, FAIL),
])
def test_the_verdict_follows_the_evidence_and_the_capacity(decide, change, verdict):
    assert decide(**change)["affordability_verdict_code"] == verdict


def test_weak_income_evidence_never_turns_a_fail_into_a_better_verdict(decide):
    out = decide(payslip_income=None, declared_income=18_500.0, declared_living_expenses=15_900.0)
    assert out["income_evidence_tier"] == 7 and out["affordability_verdict_code"] == FAIL
    assert out["outcome_code"] == 4


def test_marginal_affordability_approves_with_conditions(decide):
    out = decide(payslip_income=None, declared_income=18_500.0, bureau_score=800.0)
    assert (out["affordability_verdict_code"], out["outcome_code"]) == (MARGINAL, 2)


def test_no_income_evidence_refers_because_the_remedy_is_evidence_not_a_decline(decide):
    out = decide(payslip_income=None)
    assert out["income_source_code"] == 0 and out["outcome_code"] == 3
