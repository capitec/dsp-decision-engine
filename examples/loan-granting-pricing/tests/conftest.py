import copy
import inspect
import json
from datetime import date
from pathlib import Path

import pytest
from decikit import ConfigurableStep, DecisionTableConfig

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


def load_document(name: str):
    document = json.loads((ROOT / "configs" / "0.1.0" / f"{name}.json").read_text())
    return DecisionTableConfig.load(document) if document["type"] == "decision_table" else ConfigurableStep.load(document)


@pytest.fixture(scope="session")
def rate_card():
    return load_document("rate_card_flex_loan")


@pytest.fixture(scope="session")
def documents(rate_card):
    """What `build` receives from `configs/0.1.0`, by parameter name."""
    return {name: rate_card if name == "rate_card_flex_loan" else load_document(name)
            for name in inspect.signature(build).parameters}


@pytest.fixture(scope="session")
def params():
    return json.loads((ROOT / "configs" / "0.1.0" / "params.json").read_text())


@pytest.fixture(scope="session")
def pipeline(documents):
    return build(**documents).compile("interpreted")


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


@pytest.fixture
def parse():
    return parse_dates
