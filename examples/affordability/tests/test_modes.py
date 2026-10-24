import pytest

from assessment import modes, verdict

NEW, INCREASE, ARRANGEMENT, SCENARIO = modes.NEW_APPLICATION, modes.LIMIT_INCREASE, modes.ARRANGEMENT, modes.SCENARIO


def declared_only(assess, mode):
    return assess(assessment_mode_code=mode, applicant1_payslip_income=None, applicant1_declared_income=18500.0)


def test_a_declared_income_alone_does_not_pass_a_new_application(assess):
    out = declared_only(assess, NEW)
    assert out["income_verification_tier"] == 4
    assert out["evidence_sufficiency_code"] == verdict.INCOME_BELOW_MINIMUM_TIER
    assert out["affordability_verdict_code"] == verdict.INDETERMINATE


@pytest.mark.parametrize("mode,minimum", [(NEW, 3), (INCREASE, 4), (ARRANGEMENT, 5), (SCENARIO, 3)])
def test_each_mode_accepts_income_evidence_down_to_its_minimum_tier(assess, mode, minimum):
    assert assess(assessment_mode_code=mode)["minimum_income_tier"] == minimum


def test_a_limit_increase_accepts_the_declared_income_a_new_application_refuses(assess):
    out = declared_only(assess, INCREASE)
    assert out["evidence_sufficiency_code"] == 0
    assert out["affordability_verdict_code"] == verdict.PASS


def test_an_arrangement_accepts_the_weakest_evidence_and_a_limit_increase_does_not(assess):
    bureau_only = dict(applicant1_payslip_income=None, applicant1_bureau_estimated_income=18500.0)
    assert assess(assessment_mode_code=ARRANGEMENT, **bureau_only)["evidence_sufficiency_code"] == 0
    assert assess(assessment_mode_code=INCREASE, **bureau_only)["evidence_sufficiency_code"] == verdict.INCOME_BELOW_MINIMUM_TIER


def test_an_unknown_mode_gets_the_new_application_rule(assess):
    assert declared_only(assess, 99)["evidence_sufficiency_code"] == verdict.INCOME_BELOW_MINIMUM_TIER


def test_the_mode_changes_the_evidence_rule_but_never_the_arithmetic(assess):
    figures = ("gross_monthly_income", "net_monthly_income", "living_expenses", "existing_obligations",
               "discretionary_income", "max_affordable_instalment")
    reference = {name: assess()[name] for name in figures}
    for mode in (INCREASE, ARRANGEMENT, SCENARIO):
        out = assess(assessment_mode_code=mode)
        assert {name: out[name] for name in figures} == reference
