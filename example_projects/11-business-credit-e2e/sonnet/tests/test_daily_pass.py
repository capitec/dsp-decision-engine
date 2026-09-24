"""05's partial one-entity re-assessment (spec 05 H5, spec 11 §5.17.2's "daily
pass" example: "One entity re-scored and one event classified, 9 400 times a
day. Published as an optimisation, not as an interface"), called against a
full 11-shaped facility record, unmodified -- SCOPE.md: "05's partial
re-assessment called for one event (the daily-pass pattern)".

This project writes no code of its own here: it is `business_nested.
counterfactual.partial_reassess_entity`, reused exactly as project 05
publishes it and already proves equivalent to a full re-run in 05's own test
suite. This test exists to prove the *same* equivalence holds when called
against this project's own request shape (facility_id, existing_accounts and
the rest of this project's additions alongside 05's fields), not to re-derive
the equivalence proof itself."""
from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path

from decider import Engine

from business_nested import counterfactual

_HERE = Path(__file__).resolve().parent.parent


def _typed_request() -> dict:
    record = json.loads((_HERE / "sample_request.json").read_text())
    record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
    record["knowledge_date"] = datetime.date.fromisoformat(record["knowledge_date"])
    for account in record.get("existing_accounts", []):
        if account.get("opened_date"):
            account["opened_date"] = datetime.date.fromisoformat(account["opened_date"])
    for entity in record["entities"]:
        for event in entity.get("adverse_events", []):
            event["event_date"] = datetime.date.fromisoformat(event["event_date"])
    return record


def test_daily_pass_partial_reassessment_matches_a_full_rerun_on_an_11_shaped_record():
    build = counterfactual._load_own_pipeline_build()
    engine = Engine().bind(build())
    record = _typed_request()
    original_result = engine.score(record, {})

    refreshed_entity = copy.deepcopy(record["entities"][0])  # entity_id 1, PERSON-MDLAMINI
    assert refreshed_entity["entity_id"] == 1
    refreshed_entity["adverse_events"][0]["is_satisfied"] = True  # the daily pass's "one event changed"

    partial = counterfactual.partial_reassess_entity(record, 1, refreshed_entity, original_result)

    full_record = copy.deepcopy(record)
    full_record["entities"] = [
        refreshed_entity if e["entity_id"] == 1 else e for e in full_record["entities"]
    ]
    full_result = engine.score(full_record, {})

    assert list(partial["entity_verdict_code"]) == list(full_result["entity_verdict_code"])
    assert partial["people_grade"] == full_result["people_grade"]
    assert partial["risk_grade"] == full_result["risk_grade"]
    assert partial["business_decline_from_entity"] == full_result["business_decline_from_entity"]
