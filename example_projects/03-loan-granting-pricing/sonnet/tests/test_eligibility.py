"""Stage 5.1 eligibility gates: every gate evaluated past the first failure,
and the third `not_evaluated` state (spec 03 §5.1)."""
from __future__ import annotations

from datetime import date

from loan_granting.eligibility import FAIL, NOT_EVALUATED, PASS, _evaluate_gates

BASE = {
    "channel_code": 2, "requested_amount": 50_000.0, "decision_date": date(2026, 9, 24),
    "applicant_age_years": 38.0, "requested_term_months": 60, "has_contractual_capacity": True,
    "residency_code": 1, "employment_type_code": 1, "debt_review_status_code": 0,
    "administration_order_flag": False, "insolvency_status_code": 0, "deceased_flag": False,
    "estate_flag": False, "exclusion_list_hits": [], "is_duplicate_application": False,
    "in_flight_application_count": 0, "insolvency_rehabilitated_months": None,
}


def test_a_clean_application_passes_all_14_gates():
    out = _evaluate_gates(BASE)
    assert out["is_eligible"] is True
    assert out["eligibility_decline_reasons"] == []
    assert len(out["eligibility_gate_verdicts"]) == 14
    assert all(v == PASS for v in out["eligibility_gate_verdicts"])


def test_every_gate_is_evaluated_even_after_the_first_failure():
    """A client who fails four gates is entitled to know about all four (§5.1)."""
    app = {**BASE, "applicant_age_years": 15.0, "residency_code": 5, "deceased_flag": True,
           "debt_review_status_code": 2}
    out = _evaluate_gates(app)
    assert out["is_eligible"] is False
    assert len(out["eligibility_decline_reasons"]) == 4
    # every gate still has a verdict, not just the first failure
    assert len(out["eligibility_gate_verdicts"]) == 14
    assert out["eligibility_gate_verdicts"].count(FAIL) == 4


def test_a_gate_with_no_data_to_test_is_not_evaluated_not_passed_or_failed():
    app = {**BASE, "applicant_age_years": None}
    out = _evaluate_gates(app)
    from loan_granting.eligibility import GATE_MIN_AGE, GATE_ORDER
    idx = GATE_ORDER.index(GATE_MIN_AGE)
    assert out["eligibility_gate_verdicts"][idx] == NOT_EVALUATED


def test_exclusion_list_sentinel_zero_means_no_hit():
    app = {**BASE, "exclusion_list_hits": [0]}
    out = _evaluate_gates(app)
    assert out["is_eligible"] is True


def test_a_real_exclusion_hit_declines():
    app = {**BASE, "exclusion_list_hits": [1]}
    out = _evaluate_gates(app)
    assert out["is_eligible"] is False
    assert 1160 in out["eligibility_decline_reasons"]  # R_EXCLUSION_SANCTIONS
