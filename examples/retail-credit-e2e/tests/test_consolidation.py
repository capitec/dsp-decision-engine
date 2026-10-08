"""Consolidation: a client who fails affordability on their actual debts may pass it if they settle the external ones."""
import pytest

PASS, FAIL = 1, 3
ACTUAL, HYPOTHETICAL = 1, 2

OVERLOADED = {
    "bureau_accounts": [
        {"account_type_code": 10, "balance": 380_000.0, "limit": 400_000.0, "instalment": 4_800.0, "months_in_arrears": 0,
         "opened_date": "2020-01-01", "closed": False, "is_internal": False},
        {"account_type_code": 20, "balance": 60_000.0, "instalment": 5_500.0, "months_in_arrears": 0,
         "opened_date": "2019-05-01", "closed": False, "is_internal": False}],
    "internal_accounts": [
        {"account_type_code": 1, "balance": 5_000.0, "instalment": 300.0, "months_in_arrears": 0,
         "opened_date": "2021-06-15", "closed": False, "is_internal": True}],
    "requested_amount": 30_000.0, "term_months": 36, "is_consolidation_eligible": True,
}


def test_an_eligible_client_who_fails_on_actual_debts_is_offered_the_settlement_scenario(decide):
    out = decide(**OVERLOADED)
    assert (out["affordability_verdict_code"], out["max_affordable_instalment"]) == (FAIL, 0.0)
    assert out["existing_obligations"] == 17_800.0 and out["existing_obligations_hypothetical"] == 300.0  # internal debt stays
    assert (out["consolidation_proposed"], out["affordability_verdict_code_hypothetical"]) == (True, PASS)
    assert out["max_affordable_instalment_hypothetical"] == 2329.97


def test_the_offer_is_solved_and_validated_against_the_hypothetical_basis_and_carries_conditions(decide):
    out = decide(**OVERLOADED)
    assert (out["offer_basis_code"], out["instalment_ceiling"]) == (HYPOTHETICAL, out["instalment_cap_hypothetical"])
    assert (out["offer_amount"], out["offer_binding_constraint"], out["has_offer"]) == (30_000.0, "BIND-REQ", True)
    assert out["validation_passed"] is True and out["decline_reason_codes"] == []
    assert out["outcome_code"] == 2  # approved on condition that the external debts are settled


def test_the_actual_figures_are_kept_beside_the_hypothetical_ones(decide):
    out = decide(**OVERLOADED)
    assert out["affordability_verdict_code"] == FAIL and out["max_affordable_instalment"] == 0.0
    assert out["instalment"] > out["max_affordable_instalment"]  # the offer would fail on the actual basis


def test_a_client_who_passes_on_actual_debts_never_enters_the_scenario(decide):
    out = decide(is_consolidation_eligible=True)
    assert out["consolidation_proposed"] is False and out["offer_basis_code"] == ACTUAL
    assert out["existing_obligations_hypothetical"] == out["existing_obligations"]


def test_a_client_who_is_not_consolidation_eligible_is_declined_for_affordability(decide):
    out = decide(**{**OVERLOADED, "is_consolidation_eligible": False})
    assert out["consolidation_proposed"] is False and out["offer_basis_code"] == ACTUAL
    assert (out["has_offer"], out["decline_reason_codes"], out["outcome_code"]) == (False, [2140], 4)


def test_a_fraud_refer_suppresses_the_scenario(decide):
    out = decide(**OVERLOADED, identity_number_reused=True, synthetic_identity_score=0.9)
    assert out["fraud_verdict_code"] == 2 and out["consolidation_proposed"] is False and out["has_offer"] is False


def test_a_scenario_that_still_fails_affordability_leaves_the_decline_standing(decide):
    heavy = [{**OVERLOADED["internal_accounts"][0], "instalment": 16_000.0, "balance": 800_000.0}]
    out = decide(**{**OVERLOADED, "internal_accounts": heavy})
    assert out["consolidation_proposed"] is True and out["affordability_verdict_code_hypothetical"] == FAIL
    assert (out["offer_basis_code"], out["has_offer"], out["outcome_code"]) == (ACTUAL, False, 4)
