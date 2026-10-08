import copy
import json
from datetime import date
from pathlib import Path

import pytest

from retail_credit.pipeline import build

ROOT = Path(__file__).resolve().parents[1]


def parse_dates(record: dict) -> dict:
    """What the handler does to a JSON request before the pipeline sees it."""
    record = copy.deepcopy(record)
    record["decision_date"] = date.fromisoformat(record["decision_date"])
    for account in record["bureau_accounts"] + record["internal_accounts"]:
        account["opened_date"] = date.fromisoformat(account["opened_date"])
    return record


@pytest.fixture(scope="session")
def params():
    return json.loads((ROOT / "configs" / "0.1.0" / "params.json").read_text())


@pytest.fixture(scope="session")
def pipeline():
    return build().compile("interpreted")


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
