import copy
import json
from datetime import date
from pathlib import Path

import pytest
from decider import DecisionTableConfig

from consolidation.pipeline import build

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("rate_card_product11", "card_promotional_rates", "card_reversion_rates", "settleability_rules")


def parse_dates(record: dict) -> dict:
    """What the handler does to a JSON request before the pipeline sees it."""
    record = copy.deepcopy(record)
    for field in ("decision_date", "bureau_as_of_date", "last_consolidation_date", "last_balance_transfer_date"):
        if isinstance(record.get(field), str):
            record[field] = date.fromisoformat(record[field])
    for account in record["accounts"]:
        for field in ("opened_date", "quotation_expiry_date"):
            if field in account:
                account[field] = date.fromisoformat(account[field])
    return record


def with_params(params: dict, step: str, **changes) -> dict:
    """A copy of the params document with `changes` on the named step. Never edit one in place after scoring with it."""
    changed = copy.deepcopy(params)
    changed["consolidation"][step].update(changes)
    return changed


@pytest.fixture(scope="session")
def documents():
    return [DecisionTableConfig.load(json.loads((ROOT / "configs" / "0.1.0" / f"{name}.json").read_text())) for name in DOCUMENTS]


@pytest.fixture(scope="session")
def params():
    return json.loads((ROOT / "configs" / "0.1.0" / "params.json").read_text())


@pytest.fixture(scope="session")
def pipeline(documents):
    return build(*documents).compile("interpreted")


@pytest.fixture
def sample():
    return json.loads((ROOT / "sample_request.json").read_text())


@pytest.fixture
def assess(pipeline, sample, params):
    """`assess(**changes)` scores the sample request with the changes applied; `None` removes a field."""

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
def account(sample):
    """`account(ref, **fields)`: a copy of the sample's account with those fields changed."""

    def make(ref, **fields):
        base = next(a for a in sample["accounts"] if a["account_ref"] == ref)
        return {**copy.deepcopy(base), **fields}

    return make
