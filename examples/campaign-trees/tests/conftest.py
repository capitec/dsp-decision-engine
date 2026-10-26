import copy
import json
from datetime import date
from pathlib import Path

import pytest
from decikit import ConfigurableStep, DecisionTableConfig, TreeConfig

from campaign_trees.pipeline import build

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "0.1.0"


@pytest.fixture(scope="session")
def tree_document():
    return json.loads((CONFIGS / "campaign_23.json").read_text())


@pytest.fixture(scope="session")
def tree(tree_document):
    return TreeConfig.load(tree_document)


@pytest.fixture(scope="session")
def rate_card():
    return DecisionTableConfig.load(json.loads((CONFIGS / "rate_card_flex_loan.json").read_text()))


@pytest.fixture(scope="session")
def cap_reductions():
    return ConfigurableStep.load(json.loads((CONFIGS / "cap_reductions.json").read_text()))


@pytest.fixture(scope="session")
def params():
    return json.loads((CONFIGS / "params.json").read_text())


@pytest.fixture(scope="session")
def pipeline(tree, rate_card, cap_reductions):
    return build(tree, rate_card, cap_reductions).compile("interpreted")


@pytest.fixture
def sample():
    return json.loads((ROOT / "sample_request.json").read_text())


@pytest.fixture
def decide(pipeline, sample, params):
    """`decide(**changes)` scores the sample request with the changes applied, its date parsed as the handler does."""

    def run(params_document=None, **changes):
        record = {**sample, **changes}
        record["cycle_date"] = date.fromisoformat(record["cycle_date"])
        return pipeline.run(record, copy.deepcopy(params_document or params))

    return run
