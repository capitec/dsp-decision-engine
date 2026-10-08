import copy
import json
from pathlib import Path

import pytest

from business_nested.pipeline import build
from business_nested.records import parse_dates

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
