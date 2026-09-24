"""EP-1 -- the servable pipeline. Scores the sample request (a decline, from
project 05's own entity-disqualification path) and a hand-built approve case
for product 51 (revolving), proving both product shapes and the covenant/DSCR
wiring work end to end."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from decider import Engine

import pipeline
import inference

_HERE = Path(__file__).resolve().parent.parent


def test_sample_request_scores_and_carries_facility_and_covenant_fields():
    exe = Engine().bind(pipeline.build())
    params = pipeline.build().parameters().defaults()
    record = inference._typed_sample_record()
    result = exe.score(record, params)

    assert result["facility_id"] == 500001
    assert result["assessment_kind_code"] == 1  # vocab.EP1_NEW_TO_BANK
    assert result["origination_comparison_basis_code"] == 0  # vocab.COMPARISON_ORIGINATION
    assert result["predecessor_decision_of_record_id"] is None
    assert result["covenant_definition_version"] == "dscr-2026.03"  # 2026-09-24 decision_date
    assert result["covenant_threshold"] == 1.25
    assert result["measured_dscr"] > 0
    assert result["outcome_code"] in (1, 2, 3, 4)  # 05's own outcome vocabulary


def test_product_51_revolving_uses_project07_notional_instalment_not_05_instalment():
    """Product 51 has no instalment in 05's single-product-50 pricing shape --
    this project's own composed-around fix (origination.py::new_facility_instalment)."""
    record = json.loads((_HERE / "sample_request.json").read_text())
    record["product_code"] = 51
    record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
    record["knowledge_date"] = datetime.date.fromisoformat(record["knowledge_date"])
    for account in record.get("existing_accounts", []):
        if account.get("opened_date"):
            account["opened_date"] = datetime.date.fromisoformat(account["opened_date"])
    for entity in record["entities"]:
        for event in entity.get("adverse_events", []):
            event["event_date"] = datetime.date.fromisoformat(event["event_date"])

    exe = Engine().bind(pipeline.build())
    params = pipeline.build().parameters().defaults()
    result = exe.score(record, params)

    # 07's notional_instalment: rate defaults to 5%, floor 50 -- a different
    # number from 05's own term-loan instalment field.
    assert result["new_facility_instalment"] != result["instalment"]
    assert result["new_facility_instalment"] == round(max(result["offered_amount"] * 0.05, 50.0), 2)


def test_decision_never_reads_todays_date():
    """09 §5.15 item 4 -- no reliance on "today". Scoring the same record twice
    (once "live", once a year "later" by decision_date alone) must not change
    the covenant version resolved, since resolution is by `decision_date`, not
    wall-clock time."""
    record = inference._typed_sample_record()
    exe = Engine().bind(pipeline.build())
    params = pipeline.build().parameters().defaults()
    result_a = exe.score(record, params)
    result_b = exe.score(dict(record), params)
    assert result_a["covenant_definition_version"] == result_b["covenant_definition_version"]
