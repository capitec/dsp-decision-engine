import json
from datetime import date
from pathlib import Path

import pytest
from decikit import ConfigurableStep, DecisionTableConfig

from fraud_interdiction.pipeline import build
from fraud_interdiction.rules import RuleSet

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "0.1.0"
REASON_BASE = {"CF": 1000, "AT": 2000, "MS": 3000, "FP": 4000, "AA": 5000}
ALWAYS = {"op": ">", "feature": "amount", "threshold": 0.0}


@pytest.fixture(scope="session")
def params():
    return json.loads((CONFIGS / "params.json").read_text())


@pytest.fixture(scope="session")
def wording():
    return DecisionTableConfig.load(str(CONFIGS / "client_wording.json"))


@pytest.fixture(scope="session")
def documents():
    """The shipped documents `build` takes besides the rule set and the wording, by argument name."""
    names = ("rule_set_versions", "amount_overlays", "action_escalation", "reason_codes")
    return {name: ConfigurableStep.load(str(CONFIGS / f"{name}.json")) for name in names}


@pytest.fixture
def sample():
    """The shipped sample request with its date parsed, as the handler hands it to the pipeline."""
    record = json.loads((ROOT / "sample_request.json").read_text())
    record["decision_date"] = date.fromisoformat(record["decision_date"])
    return record


@pytest.fixture(scope="session")
def rule():
    """`rule("AT-0001", action_code=50)`: a live rule that fires on any payment, unless overridden."""

    def make(rule_id, **fields):
        family = rule_id[:2]
        return {"rule_id": rule_id, "family": family, "status": "live", "severity": 3, "action_code": 40,
                "priority": 100, "reason_code": REASON_BASE[family] + 1, "effective_from": "2025-01-01",
                "condition": ALWAYS, **fields}

    return make


@pytest.fixture(scope="session")
def decide(wording, params, documents):
    """`decide(rules, record, **overrides)`: one record through a pipeline of just these rules, interpreted."""

    def run(rules, record, stack_enabled=True, **overrides):
        rule_set = RuleSet.load({"type": "rule_set", "name": "rules", "rules": rules})
        settings = {"fraud_interdiction": {"adjustment_stack_enabled": {"adjustment_stack_enabled": stack_enabled}}}
        return build(rule_set, wording, **documents).compile("interpreted").score({**record, **overrides}, settings)

    return run
