"""Unit-level coverage for the application-level stages (scoring, financial,
grade, pricing) that don't need a ragged Python loop of their own."""
from datetime import date

from decider import Engine

from business_nested import scoring
from business_nested.financial import build_financial_unit
from business_nested.grade import build_grade_unit
from business_nested.pricing import build_pricing_unit


def test_entity_scoring_reuses_calibration_by_segment_and_applies_its_own_overlay():
    """Spec 05 §5.7: two segments through one `core.calibration` table; the
    natural-person entity overlay (segment-scoped) must not touch the juristic one."""
    r = scoring.score_entities(
        entity_id=[1, 2], entity_is_natural_person=[True, False],
        entity_bureau_score=[700.0, 700.0], entity_months_on_record=[80.0, 80.0],
        entity_worst_delinquency_months=[0.0, 0.0], entity_effective_ownership_pct=[50.0, 50.0],
        decision_date=date(2026, 9, 24),
    )
    assert r["entity_score"][0] != r["entity_score_unadjusted"][0]  # natural person: overlay applied
    assert r["entity_score"][1] == r["entity_score_unadjusted"][1]  # juristic: not in scope, untouched
    assert r["entity_adjustments_applied"][0] == "ADJ-05-ENT-001"
    assert r["entity_adjustments_applied"][1] == ""


def test_entity_scoring_handles_zero_entities():
    r = scoring.score_entities(
        entity_id=[], entity_is_natural_person=[], entity_bureau_score=[], entity_months_on_record=[],
        entity_worst_delinquency_months=[], entity_effective_ownership_pct=[], decision_date=date(2026, 9, 24),
    )
    assert r["entity_score"] == []


def test_financial_unit_computes_the_three_ratios():
    eng = Engine().bind(build_financial_unit())
    out = eng.score({
        "ebitda": 500_000.0, "finance_charges": 100_000.0, "current_assets": 800_000.0,
        "current_liabilities": 400_000.0, "interest_bearing_debt": 600_000.0, "tangible_net_worth": 900_000.0,
    }, {})
    assert out["interest_cover"] == 5.0
    assert out["current_ratio"] == 2.0
    assert abs(out["gearing"] - (600_000.0 / 900_000.0)) < 1e-9
    assert out["financial_confidence_code"] == "high"


def test_grade_unit_blends_financial_and_people_on_log_odds():
    eng = Engine().bind(build_grade_unit())
    out = eng.score({
        "financial_pd": 0.05, "people_pd": 0.05, "product_code": 51,  # revolving: no overlay scope
        "decision_date": date(2026, 6, 1), "adjustment_stack_enabled": True,
    }, {})
    # Equal inputs, equal weights sum -> the blend should reproduce the same PD.
    assert abs(out["probability_of_default_before_overlay"] - 0.05) < 1e-6
    assert out["adjustments_applied"] == []  # product 51 is out of the overlay's declared scope


def test_pricing_unit_caps_the_offer_at_the_grade_appetite():
    eng = Engine().bind(build_pricing_unit())
    out = eng.score({
        "risk_grade": 12, "security_type_code": 4, "requested_amount": 5_000_000.0,
        "requested_term_months": 36,
    }, {})
    assert out["offered_amount"] == 0.0  # grade 12's appetite is nil
    assert out["binding_constraint_code"] == "appetite_grade_cap"
