import copy
import json
from datetime import date
from pathlib import Path

import pytest

from decider import DecisionTableConfig

from collections_treatment.pipeline import build

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_LISTS = ("applicant1_bureau_accounts", "applicant1_internal_accounts",
                 "applicant2_bureau_accounts", "applicant2_internal_accounts")


def parse_dates(record: dict) -> dict:
    """What the handler does to a JSON request before the pipeline sees it."""
    record = copy.deepcopy(record)
    for field, value in record.items():
        if field.endswith("_date") and isinstance(value, str):
            record[field] = date.fromisoformat(value)
    for accounts in ACCOUNT_LISTS:
        for account in record.get(accounts, []):
            account["opened_date"] = date.fromisoformat(account["opened_date"])
    return record


@pytest.fixture(scope="session")
def matrix_document():
    return json.loads((ROOT / "configs" / "0.1.0" / "treatment_matrix.json").read_text())


@pytest.fixture(scope="session")
def pipeline(matrix_document):
    return build(DecisionTableConfig.load(matrix_document)).compile("interpreted")


@pytest.fixture(scope="session")
def params():
    return json.loads((ROOT / "configs" / "0.1.0" / "params.json").read_text())


@pytest.fixture
def sample():
    return json.loads((ROOT / "sample_request.json").read_text())


@pytest.fixture
def decide(pipeline, sample, params):
    """`decide(**changes)` scores the sample account with the changes applied; `None` removes a field."""

    def run(base=None, *, params_document=None, **changes):
        record = copy.deepcopy(base or sample)
        for field, value in changes.items():
            if value is None:
                record.pop(field, None)
            else:
                record[field] = value
        return pipeline.score(parse_dates(record), params_document or params)

    return run


@pytest.fixture
def parse():
    return parse_dates
