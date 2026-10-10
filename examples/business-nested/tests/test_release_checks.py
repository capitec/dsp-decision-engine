"""What a release check sees of the decision: the overlays and reason codes are documents, so the diff compares them."""
import json
import warnings
from datetime import date
from pathlib import Path

import pytest
from decikit import ConfigurableStep, Engine, HiddenStateWarning
from decikit.lab import diff

from business_nested import events, vocab
from business_nested.pipeline import build

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "0.2.0"
EVENT_OVERLAY = "business_credit/entities/entities/entity/events/event/overlaid_thresholds"
SCORE_OVERLAY = "business_credit/entities/entities/entity/scoring/apply_entity_score_adjustments"
PD_OVERLAY = "business_credit/decision/business_grade/apply_probability_of_default_adjustments"
RANKING = "business_credit/decision/outcome/rank_reasons"


def rebuilt(**edits):
    """`build` from the shipped documents, each named one passed through its edit first."""
    names = ("entity_score_adjustments", "business_pd_adjustments", "event_threshold_overlays", "reason_codes")
    documents = {}
    for name in names:
        document = json.loads((CONFIGS / f"{name}.json").read_text())
        edits.get(name, lambda document: None)(document)
        documents[name] = ConfigurableStep.load(document)
    return build(**documents)


def changed_paths(edited):
    """The steps whose behaviour differs from the shipped pipeline, other than the embedded affordability assessment.

    That assessment still closes over its own registers, which no diff can compare, so two builds always differ there.
    """
    changed = diff.structure(build(), edited).changed_steps
    return [c.ref.path for c in changed if "affordability_assessment" not in c.ref.path]


def test_the_business_nested_steps_read_no_hidden_objects():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Engine().bind(build(), mode="interpreted")
    flagged = [str(w.message).split(" reads ")[0] for w in caught if issubclass(w.category, HiddenStateWarning)]
    # The affordability assessment embedded in the sole-proprietor branch still closes over its own registers.
    assert all("affordability_assessment" in path for path in flagged)


def test_the_same_documents_build_an_unchanged_pipeline():
    assert changed_paths(rebuilt()) == []


def test_weakening_the_construction_overlay_changes_the_threshold_step():
    edited = rebuilt(event_threshold_overlays=lambda d: d["adjustments"][0]["effect"].update(value=0.25))
    assert changed_paths(edited) == [EVENT_OVERLAY]


def test_moving_the_construction_overlay_to_another_sector_changes_the_threshold_step():
    edited = rebuilt(event_threshold_overlays=lambda d: d["adjustments"][0]["scope"].update(sector_code=100))
    assert changed_paths(edited) == [EVENT_OVERLAY]


def test_deepening_the_new_to_bank_score_shift_changes_the_score_overlay_step():
    edited = rebuilt(entity_score_adjustments=lambda d: d["adjustments"][0]["effect"].update(value=-24.0))
    assert changed_paths(edited) == [SCORE_OVERLAY]


def test_raising_the_term_facility_odds_multiplier_changes_the_pd_overlay_step():
    edited = rebuilt(business_pd_adjustments=lambda d: d["adjustments"][0]["effect"].update(value=1.5))
    assert changed_paths(edited) == [PD_OVERLAY]


def test_changing_a_reason_severity_changes_the_ranking_step():
    edited = rebuilt(reason_codes=lambda d: d["codes"][0].update(severity_rank=99))
    assert changed_paths(edited) == [RANKING]


@pytest.mark.parametrize("document", ["entity_score_adjustments", "business_pd_adjustments", "event_threshold_overlays"])
def test_a_governance_only_edit_to_an_overlay_is_not_a_change(document):
    def reword(d):
        d["adjustments"][0].update(rationale="reworded", owner="Model Risk", approval_reference="CRC-9", review_date="2027-03-31")

    assert changed_paths(rebuilt(**{document: reword})) == []


def test_a_threshold_overlay_that_loosens_is_rejected_when_the_document_loads():
    document = json.loads((CONFIGS / "event_threshold_overlays.json").read_text())
    document["adjustments"][0]["effect"]["value"] = 2.0
    with pytest.raises(ValueError, match="tighten-only"):
        ConfigurableStep.load(document)


def test_an_overlay_for_another_target_is_rejected_when_the_document_loads():
    document = json.loads((CONFIGS / "event_threshold_overlays.json").read_text())
    document["adjustments"][0]["target"] = "score"
    with pytest.raises(ValueError, match="not for"):
        ConfigurableStep.load(document)


def test_an_overlay_on_the_disqualifying_threshold_is_applied_with_the_material_one():
    document = json.loads((CONFIGS / "event_threshold_overlays.json").read_text())
    document["adjustments"].append({**document["adjustments"][0], "adjustment_id": "ADJ-X", "target": "disqualifying_threshold"})
    out = Engine().bind(events.event_unit(ConfigurableStep.load(document)), mode="interpreted").run({
        "event_id": 1, "event_type_code": 1, "amount": 30_000.0, "event_date": date(2026, 8, 1),
        "status_is_active": True, "is_disputed": False, "is_satisfied": False,
        "decision_date": date(2026, 9, 24), "sector_code": 412, "criticality_class": vocab.CRITICAL})
    assert out["threshold_overlay_ids"] == ["ADJ-05-014", "ADJ-X"]
    assert (out["material_threshold"], out["disqualifying_threshold"]) == (5_000.0, 25_000.0)
