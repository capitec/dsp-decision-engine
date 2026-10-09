import copy
import json
from datetime import date
from pathlib import Path

import pytest

from decider import ConfigurableStep, DecisionTableConfig

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


STACKS = ("score_shift", "scaling_change", "odds_multiplier", "band_boundary_shift", "intensity_dial",
          "treatment_suppression")


@pytest.fixture(scope="session")
def stacks():
    """The shipped overlay stacks, loaded as `decider build` loads them."""
    return {name: ConfigurableStep.load(str(ROOT / "configs" / "0.1.0" / f"{name}.json")) for name in STACKS}


@pytest.fixture(scope="session")
def build_pipeline(stacks):
    """`build_pipeline(matrix_document)`: the pipeline with the shipped overlay stacks."""
    return lambda document: build(DecisionTableConfig.load(document), **stacks)


@pytest.fixture(scope="session")
def pipeline(matrix_document, build_pipeline):
    return build_pipeline(matrix_document).compile("interpreted")


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
