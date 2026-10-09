"""Everything the decision reads is in a document, a param or a step's fields, so a release check can compare it."""
import json
import warnings
from pathlib import Path

from decider import ConfigurableStep, Engine
from decider.exceptions import HiddenStateWarning
from decider.lab import diff

from campaign_trees.pipeline import build

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "0.1.0"


def cap_document(**changes):
    document = json.loads((CONFIGS / "cap_reductions.json").read_text())
    document["adjustments"][0].update(changes)
    return ConfigurableStep.load(document)


def test_building_the_pipeline_reads_no_hidden_state(tree, rate_card, cap_reductions):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Engine().bind(build(tree, rate_card, cap_reductions), mode="interpreted")
    assert [str(w.message) for w in caught if issubclass(w.category, HiddenStateWarning)] == []


def test_a_changed_cap_is_a_different_pipeline_and_a_new_rationale_is_not(tree, rate_card, cap_reductions):
    base = build(tree, rate_card, cap_reductions)
    assert diff.structure(base, build(tree, rate_card, cap_document())).identical
    assert diff.structure(base, build(tree, rate_card, cap_document(rationale="Reworded"))).identical
    assert not diff.structure(base, build(tree, rate_card, cap_document(effect={"op": "multiply", "value": 0.6}))).identical
