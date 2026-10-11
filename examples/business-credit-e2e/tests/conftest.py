import copy
import json
from pathlib import Path

import pytest

from business_credit_e2e import history, vocab
from business_credit_e2e.pipeline import build
from business_credit_e2e.records import parse_dates

ROOT = Path(__file__).resolve().parents[1]


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
    return lambda **changes: executable.run(request_for(**changes), params)


@pytest.fixture
def originate(executable, params):
    """`originate(store, request)` scores a new application and files the decision of record as the first of its facility."""

    def run(store, request):
        result = executable.run(request, params)
        return store.append(history.decision_of_record(
            request, result, assessment_kind_code=vocab.EP1_NEW_TO_BANK, comparison_basis_code=vocab.COMPARISON_ORIGINATION))

    return run
