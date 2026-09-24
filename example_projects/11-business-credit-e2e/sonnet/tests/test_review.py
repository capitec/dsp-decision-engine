"""L1 -- annual review. Builds a two-decision history (origination, then a
review a year later) and checks the four §5.4.1 outputs: grade migration with
its cause decomposition, the re-pricing decision, the limit decision (product
51, revolving), and the review-basis ladder."""
from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path

from decider import Engine

from business_credit_e2e import facility, history, review, vocab
from business_credit_e2e.reuse import load_project05_pipeline

_HERE = Path(__file__).resolve().parent.parent


def _typed_request(overrides: dict | None = None) -> dict:
    record = json.loads((_HERE / "sample_request.json").read_text())
    record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
    record["knowledge_date"] = datetime.date.fromisoformat(record["knowledge_date"])
    for account in record.get("existing_accounts", []):
        if account.get("opened_date"):
            account["opened_date"] = datetime.date.fromisoformat(account["opened_date"])
    for entity in record["entities"]:
        for event in entity.get("adverse_events", []):
            event["event_date"] = datetime.date.fromisoformat(event["event_date"])
    # A clean-enough applicant (no *disqualifying* adverse events) so the
    # facility actually originates -- the shared sample_request.json exercises
    # 05's decline path instead (see test_origination.py), which has no
    # predecessor to review. Amounts are reduced, not cleared to `[]`: an
    # entirely-empty ragged list with no non-empty sibling anywhere in the
    # record crashes single-record scoring (00/02/05 NOTES.md's own documented
    # trap -- confirmed again here).
    for entity in record["entities"]:
        for event in entity["adverse_events"]:
            event["amount"] = 500.0
    record["product_code"] = 51  # revolving -- exercises the L1 limit decision
    if overrides:
        record.update(overrides)
    return record


def _originate(request: dict) -> tuple[dict, dict]:
    project05 = load_project05_pipeline()
    exe = Engine().bind(project05.build())
    params = project05.build().parameters().defaults()
    result = exe.score(request, params)
    fac = facility.facility_shape(request["product_code"], result["offered_amount"], result.get("term_months"))
    fac["facility_id"] = request["facility_id"]
    record = history.new_decision_of_record(
        facility_id=request["facility_id"], assessment_kind_code=vocab.EP1_NEW_TO_BANK,
        decision_date=request["decision_date"], knowledge_date=request["decision_date"], predecessor_id=None,
        comparison_basis_code=vocab.COMPARISON_ORIGINATION, outcome_code=result["outcome_code"],
        risk_grade=result["risk_grade"], master_scale_version=history.master_scale_version_from_result(result),
        probability_of_default=result["probability_of_default"],
        probability_of_default_unadjusted=result["probability_of_default_unadjusted"],
        extra={"input_snapshot": request},
    )
    return record, fac


def test_annual_review_produces_all_four_5_4_1_outputs():
    store = history.DecisionHistoryStore()
    origination_request = _typed_request({"facility_id": 700001})
    previous_record, fac = _originate(origination_request)
    store.append(previous_record)
    assert previous_record["outcome_code"] in (1, 2, 3, 4)  # 05's own outcome vocabulary

    next_year_request = copy.deepcopy(origination_request)
    next_year_request["decision_date"] = datetime.date(2027, 9, 24)
    next_year_request["ebitda"] = origination_request["ebitda"] * 1.10  # business improved

    reviewed = review.annual_review(
        facility_row=fac, previous_record=previous_record, current_input=next_year_request,
        statement_age_months=6.0, management_accounts_present=True,
        contractual_rate=0.24, store=store,
    )

    assert reviewed["assessment_kind_code"] == vocab.EP3_ANNUAL_REVIEW
    assert reviewed["predecessor_decision_of_record_id"] == previous_record["decision_of_record_id"]

    migration = reviewed["grade_migration"]
    assert set(migration["causes"]) == set(vocab.CAUSE_ORDER)
    assert migration["causes_sum_to_observed"] is True

    reprice = reviewed["reprice"]
    assert reprice["action"] in (vocab.REPRICE_NONE, vocab.REPRICE_RESET_ON_NOTICE, vocab.REPRICE_FLAG_RENEGOTIATE)

    # Product 51 is revolving -- the limit decision must have run (project 07).
    assert reviewed["limit_decision"] is not None
    assert "policy_proposed_limit" in reviewed["limit_decision"]

    assert reviewed["review_basis_code"] == vocab.REVIEW_COMPLETE  # statement_age_months=6
    assert reviewed["next_review_date"] == datetime.date(2028, 9, 24)

    assert store.latest(700001)["decision_of_record_id"] == reviewed["decision_of_record_id"]
    assert len(store.history(700001)) == 2


def test_review_basis_ladder_caps_the_grade_when_stale():
    store = history.DecisionHistoryStore()
    origination_request = _typed_request({"facility_id": 700002})
    previous_record, fac = _originate(origination_request)
    store.append(previous_record)

    next_year_request = copy.deepcopy(origination_request)
    next_year_request["decision_date"] = datetime.date(2027, 9, 24)

    reviewed = review.annual_review(
        facility_row=fac, previous_record=previous_record, current_input=next_year_request,
        statement_age_months=18.0, management_accounts_present=True,  # -> STALE (§5.4.3 basis 3)
        contractual_rate=0.24, store=store,
    )
    assert reviewed["review_basis_code"] == vocab.REVIEW_STALE
    assert reviewed["risk_grade"] <= 7  # the stale-basis grade cap
    assert reviewed["next_review_date"] == datetime.date(2028, 3, 24)  # 6 months, not 12


def test_product_50_term_facility_gets_no_limit_decision():
    store = history.DecisionHistoryStore()
    origination_request = _typed_request({"facility_id": 700003, "product_code": 50})
    previous_record, fac = _originate(origination_request)
    store.append(previous_record)

    next_year_request = copy.deepcopy(origination_request)
    next_year_request["decision_date"] = datetime.date(2027, 9, 24)

    reviewed = review.annual_review(
        facility_row=fac, previous_record=previous_record, current_input=next_year_request,
        statement_age_months=6.0, management_accounts_present=True,
        contractual_rate=0.24, store=store,
    )
    assert reviewed["limit_decision"] is None  # §5.4.1 item 3 is revolving-only
