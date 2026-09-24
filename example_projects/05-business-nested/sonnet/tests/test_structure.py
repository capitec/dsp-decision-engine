from datetime import date

from business_nested import structure, vocab


def _event(event_id, amount=0.0):
    return {"event_id": event_id, "event_type_code": 1, "amount": amount, "event_date": date(2024, 1, 1),
            "status_is_active": True, "is_disputed": False, "is_satisfied": False}


def test_two_paths_to_the_same_person_are_one_entity_with_summed_ownership_and_senior_role():
    """Spec 05 §5.1, acceptance §10 item 4."""
    raw = [
        {"entity_id": 1, "entity_key": "P", "is_natural_person": True, "relationship_type_code": vocab.SHAREHOLDER,
         "effective_ownership_pct": 20.0, "is_controlling": False, "is_required_surety": False,
         "is_sole_principal": False, "adverse_events": [_event(1)]},
        {"entity_id": 1, "entity_key": "P", "is_natural_person": True, "relationship_type_code": vocab.DIRECTOR,
         "effective_ownership_pct": 15.0, "is_controlling": True, "is_required_surety": False,
         "is_sole_principal": False, "adverse_events": [_event(1)]},  # same event, second path
    ]
    r = structure.resolve_entities(raw, date(2026, 1, 1))
    assert r["resolved_entity_count"] == 1
    assert r["entity_effective_ownership_pct"] == [35.0]
    assert r["entity_relationship_type_code"] == [vocab.DIRECTOR]  # more senior than shareholder
    assert r["entity_is_controlling"] == [True]  # disjunction
    assert r["entity_path_count"] == [2]
    assert r["ev_event_id"] == [1]  # attached once, not twice


def test_resolved_count_above_bound_sets_structure_unresolved_not_a_decline():
    raw = [
        {"entity_id": i, "entity_key": f"E{i}", "is_natural_person": False,
         "relationship_type_code": vocab.GROUP_COMPANY, "effective_ownership_pct": 0.0, "is_controlling": False,
         "is_required_surety": False, "is_sole_principal": False, "adverse_events": []}
        for i in range(41)
    ]
    r = structure.resolve_entities(raw, date(2026, 1, 1))
    assert r["resolved_entity_count"] == 41
    assert r["structure_unresolved"] is True


def test_ordering_independence_shuffled_entities_and_events():
    raw = [
        {"entity_id": 1, "entity_key": "A", "is_natural_person": True, "relationship_type_code": vocab.SHAREHOLDER,
         "effective_ownership_pct": 60.0, "is_controlling": True, "is_required_surety": False,
         "is_sole_principal": False, "adverse_events": [_event(1), _event(2), _event(3)]},
        {"entity_id": 2, "entity_key": "B", "is_natural_person": True, "relationship_type_code": vocab.DIRECTOR,
         "effective_ownership_pct": 0.0, "is_controlling": False, "is_required_surety": False,
         "is_sole_principal": False, "adverse_events": []},
    ]
    r1 = structure.resolve_entities(raw, date(2026, 1, 1))
    shuffled = [dict(raw[1]), dict(raw[0])]
    shuffled[1] = dict(shuffled[1])
    shuffled[1]["adverse_events"] = list(reversed(shuffled[1]["adverse_events"]))
    r2 = structure.resolve_entities(shuffled, date(2026, 1, 1))
    assert r1["entity_id"] == r2["entity_id"]
    assert r1["ev_event_id"] == r2["ev_event_id"]


def test_criticality_classes():
    assert structure.classify_criticality(30.0, False, False, False, vocab.SHAREHOLDER) == vocab.CRITICAL
    assert structure.classify_criticality(0.0, True, False, False, vocab.DIRECTOR) == vocab.CRITICAL
    assert structure.classify_criticality(0.0, False, True, False, vocab.SURETY) == vocab.CRITICAL
    assert structure.classify_criticality(15.0, False, False, False, vocab.SHAREHOLDER) == vocab.SIGNIFICANT
    assert structure.classify_criticality(0.0, False, False, False, vocab.DIRECTOR) == vocab.SIGNIFICANT
    assert structure.classify_criticality(2.0, False, False, False, vocab.SHAREHOLDER) == vocab.PERIPHERAL
