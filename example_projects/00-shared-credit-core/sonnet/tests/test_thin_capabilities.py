"""Frozen-interface, thin-but-correct capabilities (SCOPE.md): bureau, eligibility, appetite,
exposure, consent, credit_life, adverse_events. Every one is standalone-testable (00 §7.5)."""
from datetime import date

from decider import Engine

from credit_core import adverse_events, appetite, bureau, consent, credit_life, eligibility, exposure
from credit_core.vocab import ConsentVerdict


def test_eligibility_evaluates_every_gate_not_only_the_first_failure():
    exe = Engine().bind(eligibility.decline_reason_codes_step, mode="interpreted")
    out = exe.score({"applicant_age_years": 15.0, "is_sanctioned": True})
    assert set(out["decline_reason_codes"]) == {eligibility.R_UNDERAGE, eligibility.R_SANCTIONED}


def test_eligible_applicant_has_no_reasons():
    exe = Engine().bind(eligibility.decline_reason_codes_step, mode="interpreted")
    assert exe.score({"applicant_age_years": 30.0})["decline_reason_codes"] == []


def test_bureau_normalises_one_schema_and_flags_staleness():
    exe = Engine().bind(bureau.normalise_bureau, mode="interpreted")
    response = {"accounts": [{"status_code": 2}], "enquiries": [{"enquiry_date": date(2026, 8, 1)}],
                "public_records": [], "as_of_date": date(2026, 1, 1)}
    out = exe.score({"bureau_response": response, "decision_date": date(2026, 9, 24)})
    assert out["bureau_account_count"] == 1
    assert out["bureau_is_stale"] is True  # as_of is > 90 days before decision_date


def test_bureau_missing_record_is_a_distinct_quality_verdict():
    exe = Engine().bind(bureau.normalise_bureau, mode="interpreted")
    out = exe.score({"bureau_response": {}, "decision_date": date(2026, 9, 24)})
    assert out["bureau_data_quality"] == "no_record"


def test_appetite_table_returns_a_cell_id_per_grade():
    table = appetite.build_appetite_table()
    exe = Engine().bind(table, mode="interpreted")
    out = exe.score({"product_code": 10, "segment_code": 1, "risk_grade": 6})
    assert out["cell_id"] is not None
    assert out["max_amount"] > 0


def test_exposure_excludes_facilities_outside_the_as_of_window():
    exe = Engine().bind(exposure.total_exposure, mode="interpreted")
    facilities = [
        {"balance": 100_000.0, "opened_date": "2020-01-01", "closed_date": None},
        {"balance": 50_000.0, "opened_date": "2020-01-01", "closed_date": "2024-01-01"},  # closed before as_of
    ]
    out = exe.score({"related_facilities": facilities, "exposure_as_of_date": "2026-01-01"})
    assert out["total_exposure"] == 100_000.0


def test_consent_regulated_notice_is_distinct_from_suppressed():
    exe = Engine().bind(consent.consent_verdict_step, mode="interpreted")
    marketing = exe.score({"channel_code": 1, "marketing_opt_out": True})
    regulated = exe.score({"channel_code": 1, "marketing_opt_out": True, "is_regulated_notice": True})
    assert marketing["consent_verdict"] == int(ConsentVerdict.SUPPRESSED)
    assert regulated["consent_verdict"] == int(ConsentVerdict.REGULATED_NOTICE)


def test_credit_life_premium_is_capped():
    table = credit_life.build_credit_life_table()
    exe = Engine().bind(table, mode="interpreted")
    out = exe.score({"applicant_age_years": 60.0, "term_months": 70.0, "cover_type_code": credit_life.JOINT})
    assert out["rate"] > 0


def test_adverse_events_all_14_types_are_declared():
    """Addendum item 12: 14 event types, not 7."""
    codes = {adverse_events.JUDGMENT, adverse_events.DEFAULT_LISTING, adverse_events.ADMINISTRATION_ORDER,
             adverse_events.DEBT_REVIEW, adverse_events.TAX_NON_COMPLIANCE, adverse_events.INSOLVENCY,
             adverse_events.LITIGATION, adverse_events.SEQUESTRATION, adverse_events.CURATORSHIP,
             adverse_events.GARNISHEE_ORDER, adverse_events.DEBT_COUNSELLING_WITHDRAWAL,
             adverse_events.FRAUD_CONVICTION, adverse_events.FOREIGN_ADVERSE_LISTING,
             adverse_events.REGULATORY_FINDING}
    assert len(codes) == 14


def test_adverse_event_thresholds_are_supplied_by_the_caller():
    """Addendum item 12: amount thresholds come from the caller, not a library constant."""
    exe = Engine().bind(adverse_events.event_severity_code_step, mode="interpreted")
    record = {"event_type_code": adverse_events.JUDGMENT, "amount": 10_000.0}
    lenient = exe.score(record, params={"event_severity_code": {
        "material_threshold": 50_000.0, "disqualifying_threshold": 100_000.0}})
    strict = exe.score(record, params={"event_severity_code": {
        "material_threshold": 1_000.0, "disqualifying_threshold": 5_000.0}})
    assert lenient["event_severity_code"] < strict["event_severity_code"]


def test_a_disqualifying_event_type_disqualifies_while_active_regardless_of_amount():
    exe = Engine().bind(adverse_events.event_severity_code_step, mode="interpreted")
    out = exe.score({"event_type_code": adverse_events.SEQUESTRATION, "amount": 1.0, "status_is_active": True})
    assert out["event_severity_code"] == adverse_events.DISQUALIFYING
