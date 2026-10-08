import copy
import json
from datetime import date
from pathlib import Path

import pytest

from business_nested.records import parse_dates as parse_entity_dates
from business_credit_e2e import history, vocab
from business_credit_e2e.pipeline import build

ROOT = Path(__file__).resolve().parents[1]


def parse_dates(record: dict) -> dict:
    """What the handler does to a JSON request before the pipeline sees it."""
    record = parse_entity_dates(record)
    for field in ("knowledge_date", "decision_date"):
        if isinstance(record.get(field), str):
            record[field] = date.fromisoformat(record[field])
    for account in record.get("existing_accounts", []):
        account["opened_date"] = date.fromisoformat(account["opened_date"])
    return record


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
def request_for(sample):
    """`request_for(**changes)` is the sample with its dates parsed and the changes applied; a `None` removes a field."""

    def make(**changes):
        record = copy.deepcopy(sample)
        for field, value in changes.items():
            if value is None:
                record.pop(field, None)
            else:
                record[field] = value
        return parse_dates(record)

    return make


@pytest.fixture
def decide(executable, params, request_for):
    return lambda **changes: executable.score(request_for(**changes), params)


@pytest.fixture
def originate(executable, params):
    """`originate(store, request)` scores a new application and files the decision of record as the first of its facility."""

    def run(store, request):
        result = executable.score(request, params)
        return store.append(history.decision_of_record(
            request, result, assessment_kind_code=vocab.EP1_NEW_TO_BANK, comparison_basis_code=vocab.COMPARISON_ORIGINATION))

    return run
