"""After the fact: one event removed, and one entity corrected."""
import copy

import pytest
from conftest import parse_dates

from business_nested import counterfactual
from business_nested.pipeline import DECISION_OUTPUTS


@pytest.fixture
def record(sample):
    return parse_dates(sample)


def test_without_an_event_the_decision_the_event_caused_changes(executable, record):
    answer = counterfactual.single_event_counterfactual(executable, record, 41)
    assert (answer["entity_id"], answer["event_severity_code"]) == (1, 4)
    assert (answer["entity_verdict_before"], answer["entity_verdict_after"]) == (4, 3)
    assert (answer["outcome_code_before"], answer["outcome_code_after"]) == (4, 1)
    assert answer["offered_amount_before"] == answer["offered_amount_after"] == 3_500_000.0


def test_the_classification_of_the_event_is_reported_with_its_thresholds(executable, record):
    answer = counterfactual.single_event_counterfactual(executable, record, 42)
    assert (answer["material_threshold_base"], answer["material_threshold"]) == (10_000.0, 5_000.0)
    assert answer["threshold_overlay_ids"] == ["ADJ-05-014"]
    assert (answer["event_severity_code"], answer["event_severity_code_unadjusted"]) == (3, 2)


def test_an_event_that_decides_nothing_can_be_asked_about_too(executable, record):
    answer = counterfactual.single_event_counterfactual(executable, record, 51)
    assert answer["entity_id"] == 3
    assert (answer["entity_verdict_before"], answer["entity_verdict_after"]) == (3, 2)
    assert answer["outcome_code_before"] == answer["outcome_code_after"] == 4  # event 41 still declines it


def test_the_record_asked_about_is_left_as_it_was(executable, record):
    before = copy.deepcopy(record)
    counterfactual.single_event_counterfactual(executable, record, 41)
    assert record == before


def test_asking_about_an_event_that_is_not_there_is_an_error(executable, record):
    with pytest.raises(KeyError):
        counterfactual.single_event_counterfactual(executable, record, 999)


def test_the_counterfactual_uses_the_policy_it_is_given(executable, record, with_params):
    stricter = with_params(("business_credit", "decision", "outcome", "business_outcome"), refer_cutoff_grade=7)
    answer = counterfactual.single_event_counterfactual(executable, record, 41, params=stricter)
    assert (answer["risk_grade_after"], answer["outcome_code_after"]) == (7, 3)  # grade 7 is now a referral


@pytest.fixture
def original_and_corrected(executable, record, params):
    corrected = counterfactual.without_event(record, 41)  # the judgment is rescinded
    return executable.score(record, params), corrected


def test_reassessing_one_entity_gives_the_decision_a_full_run_gives(executable, original_and_corrected, params):
    original, corrected = original_and_corrected
    partial = counterfactual.reassess_entity(corrected, original, entity_id=1)
    full = executable.score(corrected, params)
    assert {name: partial[name] for name in DECISION_OUTPUTS} == {name: full[name] for name in DECISION_OUTPUTS}
    assert (original["outcome_code"], partial["outcome_code"]) == (4, 1)


def test_reassessing_one_entity_assesses_only_that_entity(monkeypatch, original_and_corrected):
    original, corrected = original_and_corrected
    scored = []
    real = counterfactual._entity_executable()

    class Spy:
        def score(self, item):
            scored.append(item["entity_id"])
            return real.score(item)

    monkeypatch.setattr(counterfactual, "_entity_executable", lambda: Spy())
    counterfactual.reassess_entity(corrected, original, entity_id=3)
    assert scored == [3] and len(original["assessed_entities"]) == 3


def test_reassessing_an_entity_that_was_not_in_the_original_is_an_error(original_and_corrected):
    original, corrected = original_and_corrected
    corrected["entities"].append({**corrected["entities"][0], "entity_id": 9, "entity_key": "NEW", "adverse_events": []})
    with pytest.raises(KeyError):
        counterfactual.reassess_entity(corrected, original, entity_id=1)
