import math

import pytest

from credit_core.income import (
    BANK_STATEMENT, BUREAU_ESTIMATED, DECLARED, EMPLOYER_CONFIRMED, NONE_ESTABLISHED, PAYSLIP, income_unit,
)

CASES = [
    # evidence, source tier, haircut, gross
    ({"employer_confirmed_income": 20000.0, "payslip_income": 18000.0}, EMPLOYER_CONFIRMED, 0.0, 20000.0),
    ({"payslip_income": 18500.0, "declared_income": 30000.0}, PAYSLIP, 0.05, 17575.0),
    ({"statement_income": 10000.0, "statement_confidence": 0.8, "declared_income": 30000.0}, BANK_STATEMENT, 0.10, 9000.0),
    ({"statement_income": 10000.0, "statement_confidence": 0.5}, BANK_STATEMENT, 0.25, 7500.0),
    ({"statement_income": 10000.0}, BANK_STATEMENT, 0.25, 7500.0),  # no confidence counts as low
    ({"declared_income": 12000.0, "bureau_estimated_income": 50000.0}, DECLARED, 0.20, 9600.0),
    ({"bureau_estimated_income": 8000.0}, BUREAU_ESTIMATED, 0.30, 5600.0),
    ({}, NONE_ESTABLISHED, 0.0, 0.0),
]


@pytest.mark.parametrize("evidence,tier,haircut,gross", CASES)
def test_the_best_evidence_tier_wins_and_carries_its_haircut(score, evidence, tier, haircut, gross):
    out = score(income_unit, evidence)
    assert (out["income_source_code"], out["income_haircut_applied"], out["gross_monthly_income"]) == (tier, haircut, gross)
    assert out["income_verification_tier"] == tier


def test_a_verified_zero_income_is_not_the_same_as_no_evidence(score):
    verified = score(income_unit, {"payslip_income": 0.0})
    assert (verified["income_source_code"], verified["gross_monthly_income"]) == (PAYSLIP, 0.0)
    assert score(income_unit, {})["income_source_code"] == NONE_ESTABLISHED


def test_haircuts_are_policy_set_in_the_params_document(score):
    params = {"income": {"income_haircut_applied": {"haircut_payslip": 0.5}}}
    assert score(income_unit, {"payslip_income": 18500.0}, params)["gross_monthly_income"] == 9250.0


def test_the_statement_confidence_threshold_is_a_param(score):
    record = {"statement_income": 10000.0, "statement_confidence": 0.6}
    assert score(income_unit, record)["income_haircut_applied"] == 0.25
    params = {"income": {"income_haircut_applied": {"statement_confidence_threshold": 0.5}}}
    assert score(income_unit, record, params)["income_haircut_applied"] == 0.10


def test_variability_is_the_coefficient_of_variation_of_variable_pay(score):
    # mean 1 450, population standard deviation sqrt(15 000) = 122.47
    assert score(income_unit, {"variable_pay_history": [1450.0, 1600.0, 1300.0]})["income_variability_ratio"] == 0.0845
    assert score(income_unit, {"variable_pay_history": [1000.0, 1000.0]})["income_variability_ratio"] == 0.0


@pytest.mark.parametrize("history", [[], [1500.0], [0.0, 0.0]])
def test_variability_is_zero_without_two_values_or_a_mean(score, history):
    assert score(income_unit, {"variable_pay_history": history})["income_variability_ratio"] == 0.0


def test_missing_evidence_is_nan_so_it_cannot_pass_for_zero():
    from credit_core.income import income_source_code

    assert income_source_code(payslip_income=math.nan) == NONE_ESTABLISHED
