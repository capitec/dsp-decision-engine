"""What a release check sees of the pipeline: the overlays and reason codes are documents, not objects."""
import json
from pathlib import Path

import pytest
from decider import ConfigurableStep
from decider.lab import diff

from loan_granting import reasons
from loan_granting.pipeline import build

ROOT = Path(__file__).resolve().parents[1]


def document(name, edit=lambda doc: None):
    doc = json.loads((ROOT / "configs" / "0.1.0" / f"{name}.json").read_text())
    edit(doc)
    return ConfigurableStep.load(doc)


def rebuilt(documents, **edits):
    """The pipeline built from the shipped documents, with `edit(doc)` applied to the named ones."""
    return build(**{**documents, **{name: document(name, edit) for name, edit in edits.items()}})


def changed(old, new):
    """The steps that differ, apart from the embedded assessment's own (the assessment is compared in its own example)
    and the two that still close over the rate card index."""
    skipped = ("/affordability_assessment/", "loan_granting/solve_terms", "loan_granting/final_validation")
    return [c for c in diff.structure(old, new).changed_steps if not any(s in c.ref.path for s in skipped)]


@pytest.fixture(scope="module")
def shipped(documents):
    return build(**documents)


def test_the_same_documents_build_an_unchanged_pipeline(shipped, documents):
    assert changed(shipped, build(**documents)) == []


def test_doubling_a_score_overlay_changes_the_overlay_step(shipped, documents):
    doubled = rebuilt(documents, score_adjustments=lambda doc: doc["adjustments"][0]["effect"].update(value=-36.0))
    steps = changed(shipped, doubled)
    assert [c.ref.path for c in steps] == ["loan_granting/scoring/apply_score_adjustments"] and steps[0].consts


def test_a_governance_only_edit_to_an_overlay_is_not_a_change(shipped, documents):
    def reword(doc):
        doc["adjustments"][0].update(rationale="reworded", owner="Credit Risk Policy", review_date="2027-03-31")

    assert changed(shipped, rebuilt(documents, probability_of_default_adjustments=reword)) == []


def test_changing_a_reason_severity_changes_the_ranking_step(shipped, documents):
    reranked = rebuilt(documents, reason_codes=lambda doc: doc["codes"][0].update(severity_rank=99))
    assert [c.ref.path for c in changed(shipped, reranked)] == ["loan_granting/decision/rank_reasons"]


def test_every_reason_code_the_product_gives_is_in_the_reason_codes_document():
    in_document = {c["code"] for c in json.loads((ROOT / "configs" / "0.1.0" / "reason_codes.json").read_text())["codes"]}
    given = {value for name, value in vars(reasons).items() if name.startswith("R_")}
    assert given <= in_document


