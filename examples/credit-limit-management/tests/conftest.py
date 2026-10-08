import copy
import json
from pathlib import Path

import pytest

from limit_mgmt.matrix import build_matrix_table
from limit_mgmt.pipeline import build
from population import ROOT, parse_dates, sample_request


@pytest.fixture(scope="session")
def params():
    return json.loads((ROOT / "configs" / "0.1.0" / "params.json").read_text())


@pytest.fixture(scope="session")
def pipeline():
    return build(build_matrix_table()).compile("interpreted")


@pytest.fixture
def sample():
    return sample_request()


@pytest.fixture
def decide(pipeline, sample, params):
    """`decide(**changes)` scores the sample request with the changes applied; `params_document` replaces the params."""

    def run(*, params_document=None, **changes):
        record = copy.deepcopy(sample) | changes
        return pipeline.score(parse_dates(record), params_document or params)

    return run


def switch_off_every_overlay(params: dict) -> dict:
    """The params document with every `adjustment_stack_enabled` false."""
    params = copy.deepcopy(params)

    def walk(node):
        for key, value in node.items():
            if key == "adjustment_stack_enabled":
                node[key] = False
            elif isinstance(value, dict):
                walk(value)

    walk(params)
    return params
