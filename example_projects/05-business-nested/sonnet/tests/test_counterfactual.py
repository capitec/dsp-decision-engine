import copy
import datetime
import json
from pathlib import Path

import pytest
from decider import Engine

from business_nested import counterfactual

SAMPLE = Path(__file__).resolve().parents[1] / "sample_request.json"


def _typed_record():
    record = json.loads(SAMPLE.read_text())
    record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
    for entity in record["entities"]:
        for event in entity["adverse_events"]:
            event["event_date"] = datetime.date.fromisoformat(event["event_date"])
    return record


@pytest.fixture(scope="module")
def engine():
    build = counterfactual._load_own_pipeline_build()
    return Engine().bind(build())


def test_single_event_counterfactual_removes_the_disqualifying_verdict(engine):
    """Spec 05 §9.4 items 1-2: the classification and the entity verdict without it.
    Entity 1 also carries event 42 (a smaller, overlay-sensitive judgment), so removing
    event 41 downgrades the verdict from disqualifying to material -- not all the way to
    clear -- which is itself the point: the counterfactual is specific to the one event
    asked about, not "would this entity have any record at all"."""
    record = _typed_record()
    result = counterfactual.single_event_counterfactual(record, 41)
    assert result["entity_verdict_before"] == 4  # disqualifying
    assert result["entity_verdict_after"] == 3   # material, driven by the remaining event 42


def test_partial_reassessment_of_one_entity_matches_a_full_rerun():
    """Spec 05 H5 / acceptance §10 item 11, and spec 11 §5.17.5's dependency
    (SCOPE.md: "Build the partial one-entity re-assessment with an equivalence
    test, because 11 depends on it")."""
    build = counterfactual._load_own_pipeline_build()
    engine = Engine().bind(build())
    record = _typed_record()
    original_result = engine.score(record, {})

    refreshed_entity = copy.deepcopy(record["entities"][3])  # entity_id 3, PERSON-VANWYK
    assert refreshed_entity["entity_id"] == 3
    refreshed_entity["bureau_score"] = 680.0
    refreshed_entity["adverse_events"][0]["is_satisfied"] = True

    partial = counterfactual.partial_reassess_entity(record, 3, refreshed_entity, original_result)

    full_record = copy.deepcopy(record)
    full_record["entities"] = [
        refreshed_entity if e["entity_id"] == 3 else e for e in full_record["entities"]
    ]
    full_result = engine.score(full_record, {})

    assert list(partial["entity_verdict_code"]) == list(full_result["entity_verdict_code"])
    assert partial["people_grade"] == full_result["people_grade"]
    assert partial["risk_grade"] == full_result["risk_grade"]
    assert partial["business_decline_from_entity"] == full_result["business_decline_from_entity"]
