"""Structure: paths become one entity per person, and an unresolvable structure is a referral."""
import random
from datetime import date

import pytest
from decikit import Engine, flow

from business_nested import structure, vocab
from business_nested.records import entity, event

resolve = Engine().bind(flow(structure.resolved_entities), mode="interpreted")


def resolved(entities, **context):
    record = {"entities": entities, "decision_date": date(2026, 9, 24), "sector_code": 100, "requested_amount": 1e6, **context}
    return resolve.score(record)["resolved_entities"]


def test_a_person_reached_by_two_paths_is_one_entity_with_summed_ownership_and_the_most_senior_role():
    director = entity(2, vocab.DIRECTOR, 0.0, is_controlling=True, entity_key="PERSON-X")
    shareholder = entity(2, vocab.SHAREHOLDER, 55.0, entity_key="PERSON-X")
    other = entity(3, vocab.SHAREHOLDER, 45.0)
    (person, _) = resolved([shareholder, other, director])
    assert (person["entity_id"], person["path_count"]) == (2, 2)
    assert person["effective_ownership_pct"] == 55.0
    assert person["relationship_type_code"] == vocab.DIRECTOR
    assert person["is_controlling"] and person["is_owner"]


def test_ownership_adds_up_across_paths():
    paths = [entity(2, vocab.SHAREHOLDER, 12.5, entity_key="K"), entity(2, vocab.UBO, 7.5, entity_key="K")]
    assert resolved(paths)[0]["effective_ownership_pct"] == 20.0


def test_an_event_reported_on_both_paths_counts_once():
    judgment = event(41, 1_000.0)
    paths = [entity(2, 5, 30.0, [judgment], entity_key="K"), entity(2, 1, 0.0, [judgment, event(42)], entity_key="K")]
    assert [e["event_id"] for e in resolved(paths)[0]["adverse_events"]] == [41, 42]


def test_the_entity_id_of_a_person_is_the_lowest_of_their_paths():
    paths = [entity(9, 5, 10.0, entity_key="K"), entity(4, 1, 0.0, entity_key="K")]
    assert [e["entity_id"] for e in resolved(paths)] == [4]


def test_entities_and_events_come_out_in_id_order_whatever_the_request_order():
    entities = [entity(i, 5, 10.0, [event(10 * i + j) for j in (3, 1, 2)]) for i in (5, 2, 9, 1)]
    baseline = resolved(entities)
    assert [e["entity_id"] for e in baseline] == [1, 2, 5, 9]
    assert [ev["event_id"] for ev in baseline[0]["adverse_events"]] == [11, 12, 13]
    random.seed(3)
    for _ in range(5):
        shuffled = random.sample(entities, len(entities))
        assert resolved(shuffled) == baseline


def test_each_entity_and_event_carries_the_context_its_own_rules_read():
    (person,) = resolved([entity(1, 5, 100.0, [event(41)])], sector_code=412, requested_amount=3.5e6)
    assert (person["decision_date"], person["requested_amount"]) == (date(2026, 9, 24), 3.5e6)
    assert person["adverse_events"][0].items() >= {"sector_code": 412, "criticality_class": vocab.CRITICAL,
                                                   "decision_date": date(2026, 9, 24)}.items()


@pytest.mark.parametrize("fields, pct, role, expected", [
    ({"is_controlling": True}, 0.0, vocab.DIRECTOR, vocab.CRITICAL),
    ({"is_required_surety": True}, 1.0, vocab.SURETY, vocab.CRITICAL),
    ({"is_sole_principal": True}, 100.0, vocab.SOLE_PROPRIETOR_PRINCIPAL, vocab.CRITICAL),
    ({}, 25.0, vocab.SHAREHOLDER, vocab.CRITICAL),
    ({}, 24.9, vocab.SHAREHOLDER, vocab.SIGNIFICANT),
    ({}, 10.0, vocab.SHAREHOLDER, vocab.SIGNIFICANT),
    ({}, 0.0, vocab.DIRECTOR, vocab.SIGNIFICANT),
    ({}, 0.0, vocab.TRUSTEE, vocab.SIGNIFICANT),
    ({}, 9.9, vocab.SHAREHOLDER, vocab.PERIPHERAL),
    ({}, 0.0, vocab.SIGNATORY, vocab.PERIPHERAL),
])
def test_criticality_follows_control_surety_ownership_and_role(fields, pct, role, expected):
    (person,) = resolved([entity(1, role, pct, **fields)])
    assert person["criticality_class"] == expected


def test_the_ownership_thresholds_of_criticality_are_params():
    exe = Engine().bind(flow(structure.resolved_entities, name="s"), mode="interpreted")
    record = {"entities": [entity(1, 5, 15.0)], "decision_date": date(2026, 9, 24), "sector_code": 1, "requested_amount": 1.0}
    assert exe.score(record)["resolved_entities"][0]["criticality_class"] == vocab.SIGNIFICANT
    strict = {"s": {"resolved_entities": {"significant_ownership_pct": 20.0}}}
    assert exe.score(record, strict)["resolved_entities"][0]["criticality_class"] == vocab.PERIPHERAL


@pytest.mark.parametrize("owned, count, unresolved", [
    (100.0, 3, False), (90.0, 3, False), (89.9, 3, True), (110.0, 3, False), (110.1, 3, True), (100.0, 41, True),
])
def test_a_structure_that_cannot_be_seen_through_is_unresolved(assess, owned, count, unresolved):
    entities = [entity(1, 5, owned)] + [entity(i, vocab.DIRECTOR) for i in range(2, count + 1)]
    out = assess(entities)
    assert out["structure_unresolved"] is unresolved
    assert out["ownership_reconciliation_pct"] == owned and out["resolved_entity_count"] == count


def test_an_unresolved_structure_is_a_referral_never_a_decline_or_an_approval(assess):
    judgment = event(41, 500_000.0)  # disqualifying on a controlling owner, but the structure comes first
    out = assess([entity(1, 5, 40.0, [judgment], is_controlling=True)])
    assert (out["outcome_code"], out["decline_reason_codes"]) == (3, [5001])
    assert out["attributing_entity_id"] is None


def test_the_entity_limit_is_a_param(assess, with_params):
    entities = [entity(1, 5, 100.0), entity(2, vocab.DIRECTOR), entity(3, vocab.DIRECTOR)]
    assert assess(entities)["structure_unresolved"] is False
    limited = with_params(("business_credit", "decision", "structure_check"), max_entities=2)
    assert assess(entities, params_document=limited)["structure_unresolved"] is True
