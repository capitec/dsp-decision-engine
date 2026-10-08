import copy
import json
from datetime import date
from pathlib import Path

import pytest
from decider import DecisionTableConfig

from loan_granting.pipeline import build

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_LISTS = ("applicant1_bureau_accounts", "applicant1_internal_accounts",
                 "applicant2_bureau_accounts", "applicant2_internal_accounts")


def parse_dates(record: dict) -> dict:
    """What the handler does to a JSON request before the pipeline sees it."""
    record = copy.deepcopy(record)
    for field in ("decision_date", "bureau_as_of_date", "bureau_consent_expires_on"):
        if isinstance(record.get(field), str):
            record[field] = date.fromisoformat(record[field])
    for accounts in ACCOUNT_LISTS:
        for account in record.get(accounts, []):
            account["opened_date"] = date.fromisoformat(account["opened_date"])
    return record


@pytest.fixture(scope="session")
def rate_card():
    return DecisionTableConfig.load(json.loads((ROOT / "configs" / "0.1.0" / "rate_card_flex_loan.json").read_text()))


@pytest.fixture(scope="session")
def params():
    return json.loads((ROOT / "configs" / "0.1.0" / "params.json").read_text())


@pytest.fixture(scope="session")
def pipeline(rate_card):
    return build(rate_card).compile("interpreted")


@pytest.fixture
def sample():
    return json.loads((ROOT / "sample_request.json").read_text())


@pytest.fixture
def decide(pipeline, sample, params):
    """`decide(**changes)` scores the sample request with the changes applied; `None` removes a field."""

    def run(base=None, *, params_document=None, **changes):
        record = copy.deepcopy(base or sample)
        for field, value in changes.items():
            if value is None:
                record.pop(field, None)
            else:
                record[field] = value
        return pipeline.score(parse_dates(record), params_document or params)

    return run
