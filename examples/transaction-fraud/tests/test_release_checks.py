"""What a release check sees of the fraud pipeline: overlays, reason codes, cut-overs and rule data are in the diff."""
import json
import warnings
from pathlib import Path

import pytest
from decider import ConfigurableStep, DecisionTableConfig, Engine, HiddenStateWarning
from decider.lab import diff

from fraud_interdiction.pipeline import build

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "0.1.0"
DOCUMENTS = ("rule_set", "rule_set_versions", "amount_overlays", "action_escalation", "reason_codes")


def rebuilt(**edits):
    """The pipeline from the shipped documents, each named one edited by the function given."""
    loaded = {}
    for name in DOCUMENTS:
        doc = json.loads((CONFIGS / f"{name}.json").read_text())
        if name in edits:
            edits[name](doc)
        loaded[name] = ConfigurableStep.load(doc)
    return build(client_wording=DecisionTableConfig.load(str(CONFIGS / "client_wording.json")), **loaded)


@pytest.fixture(scope="module")
def shipped():
    return rebuilt()


def changed_paths(a, b):
    return [c.ref.path for c in diff.structure(a, b).changed_steps]


def test_the_pipeline_reads_no_hidden_objects(shipped):
    with warnings.catch_warnings():
        warnings.simplefilter("error", HiddenStateWarning)
        Engine().bind(shipped, mode="interpreted")


def test_the_same_documents_build_an_unchanged_pipeline(shipped):
    assert diff.structure(shipped, rebuilt()).identical


def test_doubling_the_festive_multiplier_changes_the_amount_overlay_step(shipped):
    doubled = rebuilt(amount_overlays=lambda doc: doc["stack"]["adjustments"][1]["effect"].update(value=0.35))
    assert changed_paths(shipped, doubled) == ["fraud_interdiction/amount_overlays"]


def test_a_governance_only_edit_to_an_overlay_is_not_a_change(shipped):
    def reword(doc):
        doc["stack"]["adjustments"][0].update(rationale="reworded", owner="Fraud Strategy", review_date="2027-03-31")

    assert diff.structure(shipped, rebuilt(amount_overlays=reword)).identical


def test_moving_the_escalation_window_changes_the_action_step(shipped):
    moved = rebuilt(action_escalation=lambda doc: doc["adjustments"][0].update(effective_to="2026-07-01"))
    assert changed_paths(shipped, moved) == ["fraud_interdiction/resolve_action"]


def test_changing_a_reason_severity_changes_the_action_step(shipped):
    reranked = rebuilt(reason_codes=lambda doc: doc["codes"][0].update(severity_rank=99))
    assert changed_paths(shipped, reranked) == ["fraud_interdiction/resolve_action"]


def test_moving_the_rule_set_cut_over_changes_the_version_step(shipped):
    def move(doc):
        doc["versions"][0]["effective_to"] = doc["versions"][1]["effective_from"] = "2026-02-01"

    assert changed_paths(shipped, rebuilt(rule_set_versions=move)) == ["fraud_interdiction/resolve_fraud_rule_set_version"]


def test_changing_what_a_rule_asks_for_changes_the_action_step(shipped):
    def soften(doc):
        doc["rules"][0]["action_code"] = 20 if doc["rules"][0]["action_code"] != 20 else 30

    assert "fraud_interdiction/resolve_action" in changed_paths(shipped, rebuilt(rule_set=soften))


def test_changing_how_a_rule_treats_stale_data_changes_the_firing_step(shipped):
    def edit(doc):
        rule = next(r for r in doc["rules"] if r["velocity_feature"])
        rule["stale_absent"] = "suppress" if rule["stale_absent"] != "suppress" else "evaluate_false"

    paths = changed_paths(shipped, rebuilt(rule_set=edit))
    assert any(p.endswith("resolve_firing") for p in paths), paths
