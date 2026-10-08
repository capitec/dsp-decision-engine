"""A sole proprietor lends in their own name: the statutory affordability assessment applies."""
from datetime import date

import pytest
from conftest import entity

from business_nested import vocab

PRINCIPAL = entity(1, vocab.SOLE_PROPRIETOR_PRINCIPAL, 100.0, is_sole_principal=True, is_controlling=True)


def test_a_company_is_not_assessed_for_statutory_affordability(assess):
    out = assess()
    assert out["sole_proprietor_assessment_ran"] is False
    assert (out["sole_proprietor_affordability_verdict_code"], out["sole_proprietor_max_affordable_instalment"]) == (0, 0.0)


def test_a_sole_proprietor_asking_for_more_than_they_can_service_is_declined(assess):
    out = assess([PRINCIPAL], legal_form_code=1)  # R3.5m over 48 months against a R9.2m turnover business
    assert out["sole_proprietor_assessment_ran"] is True
    assert (out["sole_proprietor_affordability_verdict_code"], out["outcome_code"]) == (3, 4)
    assert out["decline_reason_codes"] == [5601]
    assert out["sole_proprietor_max_affordable_instalment"] < out["instalment"]


def test_a_sole_proprietor_asking_for_what_they_can_service_is_approved(assess):
    out = assess([PRINCIPAL], legal_form_code=1, requested_amount=1_000_000.0)
    assert out["sole_proprietor_affordability_verdict_code"] == 1
    assert (out["outcome_code"], out["decline_reason_codes"]) == (1, [])


def test_the_assessment_tests_the_quoted_instalment_against_the_principals_capacity(assess):
    out = assess([PRINCIPAL], legal_form_code=1, requested_amount=1_000_000.0)
    assert out["sole_proprietor_max_affordable_instalment"] > out["instalment"]
    margin = assess([PRINCIPAL], legal_form_code=1, requested_amount=3_500_000.0)
    assert margin["sole_proprietor_max_affordable_instalment"] == out["sole_proprietor_max_affordable_instalment"]


def test_the_principals_personal_debts_reduce_what_they_can_afford(assess):
    debt = {"account_type_code": 1, "balance": 900_000.0, "instalment": 25_000.0, "months_in_arrears": 0,
            "opened_date": date(2021, 6, 15), "closed": False, "is_internal": True}
    base = assess([PRINCIPAL], legal_form_code=1, requested_amount=1_000_000.0)
    indebted = assess([PRINCIPAL], legal_form_code=1, requested_amount=1_000_000.0,
                      principal_internal_accounts=[debt])
    # The affordability buffer holds back 35% (product 50 has no row in its grid), so R25 000 of debt costs R16 250.
    assert indebted["sole_proprietor_max_affordable_instalment"] == pytest.approx(
        base["sole_proprietor_max_affordable_instalment"] - 25_000 * 0.65, abs=1.0)


def test_income_that_cannot_be_established_is_a_referral_not_a_decline(assess):
    out = assess([PRINCIPAL], legal_form_code=1, declared_annual_turnover=0.0)
    assert out["sole_proprietor_affordability_verdict_code"] == 4
    assert (out["outcome_code"], out["decline_reason_codes"]) == (3, [5604])


def test_the_assumed_margin_on_turnover_is_a_param(assess, with_params):
    base = assess([PRINCIPAL], legal_form_code=1, requested_amount=1_000_000.0)
    path = ("business_credit", "decision", "sole_proprietor", "statutory", "assess", "principal_income_evidence")
    thin = assess([PRINCIPAL], legal_form_code=1, requested_amount=1_000_000.0,
                  params_document=with_params(path, assumed_profit_margin=0.05))
    assert thin["sole_proprietor_max_affordable_instalment"] < base["sole_proprietor_max_affordable_instalment"]
