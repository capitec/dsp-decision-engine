"""Builders for requests: an entity, an adverse event, and the date parsing the serving handler does."""
from __future__ import annotations

import copy
from datetime import date


def event(event_id: int, amount: float | None = 1_000.0, event_type_code: int = 1, event_date: date = date(2026, 8, 1),
          **flags) -> dict:
    """An adverse event: an unsatisfied, active, undisputed judgment unless `flags` say otherwise.

    Example: `event(41, 184_000.0, is_disputed=True)`.
    """
    return {"event_id": event_id, "event_type_code": event_type_code, "amount": amount, "event_date": event_date,
            "status_is_active": True, "is_disputed": False, "is_satisfied": False, **flags}


def entity(entity_id: int, role: int = 5, pct: float = 0.0, events=(), **fields) -> dict:
    """An entity: a natural person with a good bureau record who holds `pct` percent as a shareholder (role 5).

    Example: `entity(2, role=1, is_controlling=True, entity_key="PERSON-NKOSI")` is a controlling director.
    """
    return {"entity_id": entity_id, "entity_key": f"PERSON-{entity_id}", "is_natural_person": True,
            "relationship_type_code": role, "effective_ownership_pct": pct, "is_controlling": False,
            "is_required_surety": False, "is_sole_principal": False, "bureau_score": 705.0, "months_on_record": 100.0,
            "worst_delinquency_months": 0.0, "adverse_events": list(events), **fields}


def parse_dates(record: dict) -> dict:
    """A copy of a JSON request with its dates parsed, as `Executable.run` takes them (the handler does this when serving)."""
    record = copy.deepcopy(record)
    if isinstance(record.get("decision_date"), str):
        record["decision_date"] = date.fromisoformat(record["decision_date"])
    for entity_record in record["entities"]:
        for adverse in entity_record.get("adverse_events", []):
            if isinstance(adverse["event_date"], str):
                adverse["event_date"] = date.fromisoformat(adverse["event_date"])
    return record
