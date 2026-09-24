from business_nested import rollup, vocab
from credit_core.adverse_events import DISQUALIFYING, MINOR, JUDGMENT


def _ev(event_id, severity, amount=0.0, satisfied=False, months_ago=1, event_type_code=JUDGMENT):
    return {"event_id": event_id, "severity": severity, "amount": amount, "is_satisfied": satisfied,
            "months_ago": months_ago, "event_type_code": event_type_code}


def test_count_rule_attributes_to_the_whole_satisfying_set():
    """AE-R-02, acceptance §10 item 2: a count-based rule attributes to *all* the
    events that satisfied it, not to one of them."""
    events = [_ev(1, MINOR, months_ago=2), _ev(2, MINOR, months_ago=5), _ev(3, MINOR, months_ago=8)]
    verdict, rule, attributed = rollup.rollup_one_entity(events, vocab.SIGNIFICANT, 100_000.0)
    assert verdict == vocab.MATERIAL
    assert rule == rollup.AE_R_02
    assert attributed == [1, 2, 3]


def test_any_disqualifying_event_declines_the_entity():
    events = [_ev(9, DISQUALIFYING, amount=90_000.0)]
    verdict, rule, attributed = rollup.rollup_one_entity(events, vocab.CRITICAL, 100_000.0)
    assert verdict == vocab.DISQUALIFYING and rule == rollup.AE_R_01 and attributed == [9]


def test_no_events_is_clear():
    verdict, rule, attributed = rollup.rollup_one_entity([], vocab.PERIPHERAL, 100_000.0)
    assert verdict == vocab.CLEAR


def test_peripheral_disqualifying_is_capped_at_material_unless_write_off_or_fraud():
    events = [_ev(5, DISQUALIFYING, amount=300_000.0)]
    verdict, rule, attributed = rollup.rollup_one_entity(events, vocab.PERIPHERAL, 100_000.0)
    assert verdict == vocab.MATERIAL and rule == rollup.AE_R_12


def test_aggregate_unsatisfied_amount_escalates_to_disqualifying():
    events = [_ev(1, MINOR, amount=300_000.0), _ev(2, MINOR, amount=300_000.0)]
    verdict, rule, attributed = rollup.rollup_one_entity(events, vocab.SIGNIFICANT, 100_000.0)
    assert verdict == vocab.DISQUALIFYING and rule == rollup.AE_R_07
    assert set(attributed) == {1, 2}


def test_shuffled_event_order_gives_the_same_verdict_and_attribution():
    """Acceptance §10 item 3: invariant under shuffling."""
    events = [_ev(1, MINOR, months_ago=2), _ev(2, MINOR, months_ago=5), _ev(3, MINOR, months_ago=8)]
    v1, r1, a1 = rollup.rollup_one_entity(events, vocab.SIGNIFICANT, 100_000.0)
    v2, r2, a2 = rollup.rollup_one_entity(list(reversed(events)), vocab.SIGNIFICANT, 100_000.0)
    assert (v1, r1, a1) == (v2, r2, a2)
