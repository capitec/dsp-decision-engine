import json
from datetime import date
from pathlib import Path

import pytest
from decider import ConfigurableStep, DecisionTableConfig

from credit_core.pipeline import build

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "0.1.0"


@pytest.fixture(scope="session")
def params():
    return json.loads((CONFIGS / "params.json").read_text())


@pytest.fixture(scope="session")
def score_adjustments():
    return ConfigurableStep.load(str(CONFIGS / "score_adjustments.json"))


@pytest.fixture(scope="session")
def reason_codes():
    return ConfigurableStep.load(str(CONFIGS / "reason_codes.json"))


@pytest.fixture(scope="session")
def quote_pipeline(score_adjustments, reason_codes):
    return build(DecisionTableConfig.load(str(CONFIGS / "rate_card_flex_loan.json")), score_adjustments, reason_codes)


@pytest.fixture(scope="session")
def quote(quote_pipeline):
    """The Flex Loan quote pipeline, interpreted (compiled modes are exercised in test_serving.py)."""
    return quote_pipeline.compile("interpreted")


@pytest.fixture
def sample():
    """The shipped sample request with its dates parsed, as the handler would hand them to the pipeline."""
    record = json.loads((ROOT / "sample_request.json").read_text())
    record["decision_date"] = date.fromisoformat(record["decision_date"])
    for account in record["bureau_accounts"] + record["internal_accounts"]:
        account["opened_date"] = date.fromisoformat(account["opened_date"])
    return record


@pytest.fixture(scope="session")
def score():
    """`score(step, record, params=None)`: one record through a step or flow, interpreted."""

    def run(step, record, params=None):
        return step.compile("interpreted").score(record, params)

    return run
