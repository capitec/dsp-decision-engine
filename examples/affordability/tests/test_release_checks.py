"""What a release check sees of the assessment: the capacity overlays and the reason codes are in the diff, and no step reads a hidden object."""
import json
import warnings

import pytest
from decider import ConfigurableStep, Engine, HiddenStateWarning
from decider.lab import diff

from assessment import documents
from assessment.pipeline import build


def rebuilt(overlay=lambda doc: None, reasons=lambda doc: None):
    docs = []
    for name, edit in (("capacity_adjustments", overlay), ("reason_codes", reasons)):
        doc = json.loads((documents.CONFIGS / f"{name}.json").read_text())
        edit(doc)
        docs.append(ConfigurableStep.load(doc))
    return build(*docs)


@pytest.fixture(scope="module")
def shipped():
    return build()


def test_the_assessment_reads_no_hidden_objects(shipped):
    with warnings.catch_warnings():
        warnings.simplefilter("error", HiddenStateWarning)
        Engine().bind(shipped, mode="interpreted")


def test_the_same_documents_build_an_unchanged_pipeline(shipped):
    assert diff.structure(shipped, rebuilt()).identical


def test_tightening_the_capacity_overlay_changes_the_overlay_step(shipped):
    tighter = rebuilt(lambda doc: doc["adjustments"][0]["effect"].update(value=0.80))
    changed = diff.structure(shipped, tighter).changed_steps
    assert [c.ref.path for c in changed] == [
        "affordability_assessment/capacity/overlay/apply_max_affordable_instalment_adjustments"] and changed[0].consts


def test_a_governance_only_edit_to_an_overlay_is_not_a_change(shipped):
    def reword(doc):
        doc["adjustments"][0].update(rationale="reworded", owner="Model Risk", review_date="2027-03-31")

    assert diff.structure(shipped, rebuilt(reword)).identical


def test_changing_a_reason_severity_changes_the_ranking_step(shipped):
    reranked = rebuilt(reasons=lambda doc: doc["codes"][0].update(severity_rank=99))
    changed = diff.structure(shipped, reranked).changed_steps
    assert [c.ref.path for c in changed] == ["affordability_assessment/capacity/reasons/rank_reasons"]


def test_a_tighter_overlay_lowers_the_served_capacity(sample, parse, params):
    sample = sample | {"product_code": 10}
    base = build().compile("interpreted").score(parse(sample), params)["max_affordable_instalment"]
    tighter = rebuilt(lambda doc: doc["adjustments"][0]["effect"].update(value=0.80))
    assert tighter.compile("interpreted").score(parse(sample), params)["max_affordable_instalment"] < base
