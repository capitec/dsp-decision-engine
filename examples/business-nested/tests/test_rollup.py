"""The adverse verdict of an entity: counts, aggregates, the peripheral cap, and the events that justify it."""
import random
from datetime import date

import pytest
from decider import Engine

from business_nested import rollup, vocab
from business_nested.records import entity, event

CLEAR, MINOR, MATERIAL, DISQUALIFYING = 1, 2, 3, 4
POLICY = dict(recent_months=12, minor_count=3, material_floor=150_000.0, material_pct=0.15,
              disqualifying_floor=500_000.0, disqualifying_pct=0.5, immaterial_count=2)


def judged(event_id, severity, amount=1_000.0, age_months=2, event_type_code=1, **flags):
    """A classified event as the rollup reads it."""
    return {"event_id": event_id, "event_type_code": event_type_code, "amount": amount, "age_months": age_months,
            "is_satisfied": False, "is_disputed": False, "event_severity_code": severity,
            "event_severity_code_unadjusted": severity, **flags}


def verdict(events, criticality=vocab.CRITICAL, requested_amount=1_000_000.0, key="event_severity_code"):
    return rollup.rollup(events, key, criticality, requested_amount, **POLICY)


def test_an_entity_without_events_is_clear():
    assert verdict([]) == (CLEAR, rollup.AE_R_11, [])


def test_one_disqualifying_event_disqualifies_and_names_itself():
    events = [judged(1, MINOR), judged(2, DISQUALIFYING, 300_000.0), judged(3, MATERIAL, 30_000.0)]
    assert verdict(events) == (DISQUALIFYING, "AE-R-01", [2])


def test_three_minor_events_in_a_year_are_material_and_all_three_are_named():
    events = [judged(i, MINOR, age_months=m) for i, m in ((1, 1), (2, 6), (3, 12))] + [judged(4, MINOR, age_months=13)]
    assert verdict(events) == (MATERIAL, "AE-R-02", [1, 2, 3])


def test_two_recent_minor_events_are_only_minor():
    events = [judged(1, MINOR), judged(2, MINOR), judged(3, MINOR, age_months=30)]
    assert verdict(events) == (MINOR, rollup.WORST_EVENT, [1])


def test_six_unsatisfied_judgments_that_are_each_minor_add_up_to_a_material_entity():
    events = [judged(i, MINOR, 30_000.0, age_months=20 + i) for i in range(1, 7)]
    assert verdict(events) == (MATERIAL, "AE-R-06", [1, 2, 3, 4, 5, 6])


def test_the_aggregate_limits_scale_with_the_amount_requested():
    events = [judged(i, MINOR, 30_000.0, age_months=20 + i) for i in range(1, 7)]  # 180 000 unsatisfied
    assert verdict(events, requested_amount=1_200_000.0)[:2] == (MINOR, rollup.WORST_EVENT)  # 15% is now 180 000
    assert verdict(events, requested_amount=1_199_999.0)[:2] == (MATERIAL, "AE-R-06")


def test_a_large_unsatisfied_total_disqualifies_even_when_no_event_does():
    events = [judged(i, MATERIAL, 90_000.0, age_months=20 + i) for i in range(1, 7)]
    assert verdict(events, requested_amount=800_000.0) == (DISQUALIFYING, "AE-R-07", [1, 2, 3, 4, 5, 6])


def test_satisfied_events_do_not_count_towards_the_unsatisfied_total():
    events = [judged(i, MINOR, 30_000.0, age_months=20 + i) for i in range(1, 6)]
    events.append(judged(6, MINOR, 30_000.0, age_months=26, is_satisfied=True))
    assert verdict(events)[:2] == (MINOR, rollup.WORST_EVENT)  # 150 000 does not exceed 150 000


def test_a_couple_of_immaterial_events_leave_the_entity_clear():
    assert verdict([judged(1, CLEAR), judged(2, CLEAR)]) == (CLEAR, "AE-R-11", [1, 2])
    assert verdict([judged(1, CLEAR), judged(2, CLEAR), judged(3, CLEAR)]) == (CLEAR, rollup.WORST_EVENT, [])


def test_a_peripheral_entity_is_capped_at_material():
    events = [judged(1, DISQUALIFYING, 300_000.0)]
    assert verdict(events, vocab.PERIPHERAL) == (MATERIAL, "AE-R-12", [1])
    assert verdict(events, vocab.SIGNIFICANT)[0] == DISQUALIFYING


@pytest.mark.parametrize("event_type_code", rollup.NEVER_CAPPED_EVENT_TYPES)
def test_the_cap_does_not_hide_a_sequestration_a_fraud_conviction_or_an_administration_order(event_type_code):
    events = [judged(1, DISQUALIFYING, None, event_type_code=event_type_code)]
    assert verdict(events, vocab.PERIPHERAL) == (DISQUALIFYING, "AE-R-01", [1])


def test_the_verdict_does_not_depend_on_the_order_of_the_events():
    events = [judged(i, MINOR, 30_000.0, age_months=i * 2) for i in range(1, 8)] + [judged(9, MATERIAL, 40_000.0)]
    expected = verdict(events)
    random.seed(1)
    for _ in range(10):
        assert verdict(random.sample(events, len(events))) == expected


# The same rules inside the pipeline, over events that go through classification first.
def adverse(assess, events, pct=100.0, **request):
    out = assess([entity(1, 5, pct, events)], **request)
    return out["assessed_entities"][0]


def test_an_overlay_that_decides_the_verdict_is_reported_beside_the_verdict_without_it(assess):
    judgment = [event(1, 7_000.0)]  # base material threshold 10 000, halved to 5 000 in construction
    construction = adverse(assess, judgment, sector_code=412)
    elsewhere = adverse(assess, judgment, sector_code=100)
    assert (construction["adverse_verdict_code"], construction["adverse_verdict_code_unadjusted"]) == (MATERIAL, MINOR)
    assert construction["adverse_verdict_overlay_sensitive"] is True
    assert (elsewhere["adverse_verdict_code"], elsewhere["adverse_verdict_overlay_sensitive"]) == (MINOR, False)


def test_a_verdict_that_rests_on_a_disputed_event_is_provisional(assess):
    events = [event(1, 80_000.0, is_disputed=True), event(2, 900.0)]
    out = adverse(assess, events)
    assert (out["adverse_verdict_code"], out["adverse_event_ids"], out["adverse_verdict_provisional"]) == (MATERIAL, [1], True)


def test_the_verdict_without_disputed_events_is_reported_too(assess):
    events = [event(i, 90_000.0, is_disputed=i <= 2, event_date=date(2024, i, 1)) for i in range(1, 7)]
    out = adverse(assess, events, pct=15.0, requested_amount=800_000.0)  # a significant entity: R90 000 is material
    assert out["adverse_verdict_code"] == DISQUALIFYING and out["adverse_verdict_code_undisputed"] == MATERIAL


def test_the_rule_thresholds_are_params(assess, with_params):
    events = [event(i, 1_000.0, event_date=date(2026, 8, i)) for i in (1, 2, 3)]
    assert adverse(assess, events)["adverse_binding_rule"] == "AE-R-02"
    stricter = with_params(("business_credit", "entities", "entities"), minor_count=4)
    out = assess([entity(1, 5, 100.0, events)], params_document=stricter)["assessed_entities"][0]
    assert out["adverse_verdict_code"] == MINOR
