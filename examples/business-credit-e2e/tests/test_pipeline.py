"""EP-1 as a pipeline: the assessment, the instalment the covenant is tested on, and the covenant bound at approval."""
import pytest


def test_the_sample_is_approved_and_priced_by_the_assessment(decide):
    out = decide()
    assert (out["outcome_code"], out["risk_grade"], out["offered_amount"], out["instalment"]) == (1, 7, 1_000_000.0, 30_392.91)
    assert out["master_scale_version"] == "biz-rg-2026.09"  # the version inside risk_grade_cell_id


def test_the_covenant_is_tested_on_existing_debt_service_plus_the_new_instalment(decide):
    out = decide()
    # the stated R28 000 loan instalment, and the card imputed at 3% of its R250 000 limit
    assert out["dscr_existing_debt_service"] == 35_500.0
    assert out["measured_dscr"] == pytest.approx(1_100_000 / 12 / (35_500 + 30_392.91), abs=1e-4)
    assert (out["covenant_threshold"], out["dscr_breach_class_code"]) == (1.25, 0)


def test_weak_earnings_breach_the_covenant_at_origination_and_say_how_badly(decide):
    out = decide(ebitda=400_000.0)
    assert out["measured_dscr"] == pytest.approx(0.506, abs=1e-3)
    assert out["dscr_headroom_ratio"] == pytest.approx(0.506 - 1.25, abs=1e-3)
    assert out["dscr_breach_class_code"] == 3


def test_a_revolving_facility_is_tested_on_the_minimum_payment_on_its_whole_limit(decide):
    out = decide(product_code=51, requested_amount=400_000.0)
    assert out["offered_amount"] == 400_000.0
    assert out["new_facility_instalment"] == 20_000.0  # 5% of the limit, not a term loan's instalment
    assert out["new_facility_instalment"] != out["instalment"]


def test_a_small_revolving_limit_pays_at_least_the_minimum_payment(decide):
    assert decide(product_code=51, requested_amount=15_000.0)["new_facility_instalment"] == 750.0
    assert decide(product_code=51, requested_amount=500.0)["new_facility_instalment"] == 50.0


@pytest.mark.parametrize("decision_date, version, threshold", [
    ("2025-06-01", "dscr-2024.01", 1.20), ("2026-09-24", "dscr-2026.03", 1.25), ("2029-01-01", "dscr-2028.09", 1.25)])
def test_the_covenant_version_is_the_one_in_force_on_the_decision_date(decide, decision_date, version, threshold):
    out = decide(decision_date=decision_date, knowledge_date=decision_date)
    assert (out["covenant_definition_version"], out["covenant_threshold"]) == (version, threshold)


def test_the_covenant_instance_is_named_by_the_facility_and_scoring_twice_gives_the_same_answer(decide):
    first, second = decide(), decide()
    assert first["covenant_instance_id"] == second["covenant_instance_id"] == "COV-DSCR-500001"
    assert first == second


def test_the_policy_numbers_of_the_assessment_are_in_this_projects_params_document(params, executable):
    assert "business_credit" in params["business_credit_e2e"]
    assert params["business_credit_e2e"]["new_facility_instalment"] == {"minimum_payment_pct": 0.05, "minimum_payment": 50.0}
