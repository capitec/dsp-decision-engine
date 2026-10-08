import copy
import json
from datetime import date
from pathlib import Path

import pytest

from business_nested.pipeline import build

ROOT = Path(__file__).resolve().parents[1]
DECISION_DATE = date(2026, 9, 24)


def parse_dates(record: dict) -> dict:
    """What the handler does to a JSON request before the pipeline sees it."""
    record = copy.deepcopy(record)
    if isinstance(record.get("decision_date"), str):
        record["decision_date"] = date.fromisoformat(record["decision_date"])
    for entity in record["entities"]:
        for event in entity.get("adverse_events", []):
            if isinstance(event["event_date"], str):
                event["event_date"] = date.fromisoformat(event["event_date"])
    return record


def event(event_id: int, amount: float | None = 1_000.0, event_type_code: int = 1, event_date: date = date(2026, 8, 1),
          **flags) -> dict:
    """An adverse event; an unsatisfied, active, undisputed judgment unless told otherwise."""
    return {"event_id": event_id, "event_type_code": event_type_code, "amount": amount, "event_date": event_date,
            "status_is_active": True, "is_disputed": False, "is_satisfied": False, **flags}


def entity(entity_id: int, role: int = 5, pct: float = 0.0, events=(), **fields) -> dict:
    """An entity record: a natural person with a good bureau record, owning `pct` percent as a shareholder by default."""
    return {"entity_id": entity_id, "entity_key": f"PERSON-{entity_id}", "is_natural_person": True,
            "relationship_type_code": role, "effective_ownership_pct": pct, "is_controlling": False,
            "is_required_surety": False, "is_sole_principal": False, "bureau_score": 705.0, "months_on_record": 100.0,
            "worst_delinquency_months": 0.0, "adverse_events": list(events), **fields}


@pytest.fixture(scope="session")
def params():
    return json.loads((ROOT / "configs" / "0.1.0" / "params.json").read_text())


@pytest.fixture(scope="session")
def executable():
    return build().compile("interpreted")


@pytest.fixture
def sample():
    return json.loads((ROOT / "sample_request.json").read_text())


@pytest.fixture
def assess(executable, sample, params):
    """`assess(entities=None, params_document=None, **changes)` scores the sample with the changes applied.

    `entities` replaces the entity list; a `None` value removes a field of the request.
    """

    def run(entities=None, *, params_document=None, **changes):
        record = copy.deepcopy(sample)
        if entities is not None:
            record["entities"] = entities
        for field, value in changes.items():
            if value is None:
                record.pop(field, None)
            else:
                record[field] = value
        return executable.score(parse_dates(record), params_document or params)

    return run


@pytest.fixture
def with_params(params):
    """`with_params({"path/to": {...}})` is the params document with those leaves changed: `with_params(("business_credit",
    "decision", "outcome", "business_outcome"), refer_cutoff_grade=7)`."""

    def change(path, **values):
        document = copy.deepcopy(params)
        node = document
        for key in path:
            node = node[key]
        node.update(values)
        return document

    return change
