"""The cap waterfall (spec 03 §5.5): 52 rules, attribution, the uplift exception."""
from __future__ import annotations

from loan_granting.waterfall import (
    REGISTER, SEED_AMOUNT_CAP, SEED_TERM_CAP, SEED_WORST_GRADE, UPLIFT_AUTHORITY_CEILING, _run_waterfall,
)

BASE_APP = {
    "risk_grade": 7, "channel_code": 2, "employment_type_code": 1, "internal_tenure_months": 41.0,
    "months_employed": 41.0, "worst_arrears_months": 0, "accounts_in_arrears_count": 0,
    "bureau_enquiry_count_60d": 3, "employer_id": 1, "group_exposure_limit": 180_000.0,
    "internal_exposure_total": 22_000.0, "campaign_id": None, "campaign_is_preapproved": False,
    "purpose_code": 1, "residency_code": 1, "requested_term_months": 60, "applicant_age_years": 38.0,
}


def test_register_has_52_rules_at_the_declared_ownership_split():
    assert len(REGISTER) == 52
    owners = [r["owner"] for r in REGISTER]
    assert owners.count("Credit Risk Policy") == 31
    assert owners.count("Unsecured Lending Product") == 12
    assert owners.count("Credit Systems") == 6
    assert owners.count("Financial Crime") == 3


def test_sequence_positions_are_unique_and_span_1_to_52():
    positions = sorted(r["sequence"] for r in REGISTER)
    assert positions == list(range(1, 53))


def test_every_rule_gets_a_verdict_and_final_value_is_explained():
    out = _run_waterfall(BASE_APP)
    assert len(out["waterfall_rule_ids"]) == 52
    assert len(out["waterfall_rule_status"]) == 52
    # Every ceiling's final value has a named binder unless nothing ever bound it.
    assert out["amount_cap"] <= SEED_AMOUNT_CAP
    assert out["amount_cap_binding_rule_id"] is not None
    assert out["term_cap"] <= SEED_TERM_CAP
    assert out["worst_acceptable_grade"] <= SEED_WORST_GRADE


def test_grade_12_is_declined_by_the_appetite_rule():
    out = _run_waterfall({**BASE_APP, "risk_grade": 12})
    assert 1183 in out["waterfall_decline_reason_codes"]  # R_GRADE_BEYOND_APPETITE


def test_employment_tenure_under_3_months_declines():
    out = _run_waterfall({**BASE_APP, "months_employed": 1.0})
    assert 1181 in out["waterfall_decline_reason_codes"]  # R_EMPLOYMENT_TENURE_TOO_SHORT


def test_campaign_uplift_raises_the_amount_cap_within_its_authority():
    # Note: a filler rule's own condition keys off campaign_id being present/absent, so
    # the two runs' pre-uplift base can differ slightly -- the authority bound (below) is
    # the property that must hold regardless, not equality with the no-campaign run.
    plain = _run_waterfall(BASE_APP)
    uplifted = _run_waterfall({**BASE_APP, "campaign_id": 4471, "campaign_is_preapproved": True})
    assert uplifted["amount_cap"] > plain["amount_cap"]
    assert uplifted["amount_cap"] <= UPLIFT_AUTHORITY_CEILING
    assert uplifted["uplift_authority_reference"] is not None


def test_uplift_never_exceeds_the_authority_ceiling():
    rich = {**BASE_APP, "risk_grade": 1, "campaign_id": 4471, "campaign_is_preapproved": True}
    out = _run_waterfall(rich)
    from loan_granting.waterfall import UPLIFT_AUTHORITY_CEILING
    assert out["amount_cap"] <= UPLIFT_AUTHORITY_CEILING


def test_a_not_applicable_rule_is_distinct_from_a_rule_that_did_not_bind():
    out = _run_waterfall(BASE_APP)
    statuses = set(out["waterfall_rule_status"])
    assert "not_applicable" in statuses
    assert "evaluated_did_not_bind" in statuses


def test_pensioner_is_capped_at_80000():
    out = _run_waterfall({**BASE_APP, "employment_type_code": 4, "risk_grade": 3})
    assert out["amount_cap"] <= 80_000.0


def test_arrears_history_tightens_both_amount_and_grade():
    clean = _run_waterfall(BASE_APP)
    arrears = _run_waterfall({**BASE_APP, "worst_arrears_months": 3, "accounts_in_arrears_count": 1})
    assert arrears["amount_cap"] < clean["amount_cap"]
    assert arrears["worst_acceptable_grade"] < clean["worst_acceptable_grade"]
