"""Entity scoring: a scorecard, a tighten-only overlay, calibration to a probability of default, and a grade.

`entity_scoring_unit()` is the flow `each` runs on every entity, so each entity carries its own score,
probability of default and grade, beside the unadjusted ones. The grade ladder is also used by the people
and financial components, which grade a blended probability the same way.
"""
from __future__ import annotations

from datetime import date

from decider import DecisionTableConfig, ScorecardConfig, flow, step

from credit_core import calibration
from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.evidence import cell_id
from credit_core.vocab import SEGMENT_SME_ENTITY, SEGMENT_SME_PEOPLE

GRADE_VERSION = "bus-rg-2026.09"
ENTITY_ADJUSTMENT_SET_ID = "AS-05-ENT-2026.09"

# segment -> the 11 ascending probability-of-default cut points between grades 1 (best) and 12.
_GRADE_CUTS = {
    SEGMENT_SME_PEOPLE: [0.01, 0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.25, 0.34, 0.45, 0.60],
    SEGMENT_SME_ENTITY: [0.008, 0.018, 0.032, 0.05, 0.075, 0.11, 0.155, 0.21, 0.29, 0.39, 0.52],
}

# A natural person behind a new-to-bank business scores 12 points lower: defaults above the model's prediction.
ENTITY_OVERLAYS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-05-ENT-001", kind="score_shift", target="entity_score", effect=AdjustmentEffect("add", -12.0),
        scope={"segment_code": SEGMENT_SME_PEOPLE}, stack_position=2, owner="Business Credit Risk Policy",
        approval_reference="CRC-2026-041",
        rationale="Natural-person entities behind new-to-bank businesses defaulting above prediction",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True),
])


def build_entity_scorecard() -> ScorecardConfig:
    """Four characteristics: bureau score, months on record, worst delinquency and ownership stake.

    Writes `score_before_overlay`. The bureau cut-offs are params; a missing characteristic scores its default bin.
    """
    def edge(name, default):
        return {"param": name, "default": default}

    return ScorecardConfig.load({
        "type": "scorecard", "name": "entity_scorecard", "output_name": "score_before_overlay",
        "variables": [
            {"type": "constant", "score": 600, "output_name": "entity_scorecard_offset"},
            {"type": "scored", "variable_name": "bureau_score", "strict": False,
             "default": {"value": -20, "name": "no_bureau"}, "bins": [
                 {"value": -35, "upper_bound": edge("bureau_low", 500.0)},
                 {"value": -5, "lower_bound": edge("bureau_low", 500.0), "upper_bound": edge("bureau_mid", 650.0)},
                 {"value": 20, "lower_bound": edge("bureau_mid", 650.0), "upper_bound": edge("bureau_high", 750.0)},
                 {"value": 40, "lower_bound": edge("bureau_high", 750.0)}]},
            {"type": "scored", "variable_name": "months_on_record", "strict": False,
             "default": {"value": -10, "name": "unknown"}, "bins": [
                 {"value": -20, "upper_bound": 12.0}, {"value": 0, "lower_bound": 12.0, "upper_bound": 60.0},
                 {"value": 20, "lower_bound": 60.0}]},
            {"type": "scored", "variable_name": "worst_delinquency_months", "strict": False,
             "default": {"value": 0, "name": "no_history"}, "bins": [
                 {"value": 25, "upper_bound": 0.5}, {"value": -5, "lower_bound": 0.5, "upper_bound": 3.0},
                 {"value": -30, "lower_bound": 3.0}]},
            {"type": "scored", "variable_name": "effective_ownership_pct", "strict": False,
             "default": {"value": 0, "name": "unknown"}, "bins": [
                 {"value": -5, "upper_bound": 10.0}, {"value": 0, "lower_bound": 10.0, "upper_bound": 50.0},
                 {"value": 10, "lower_bound": 50.0}]},
        ],
    })


def grade_lookup(probability: str, grade: str, segment: str = "segment_code") -> DecisionTableConfig:
    """Grade 1 (best) to 12 for the probability of default in column `probability` and the segment in column `segment`.

    The segment is 4 (people) or 5 (juristic); any other gets grade 12. Writes `grade` and `<grade>_cell_id`.

    Example: `grade_lookup("people_pd", "people_grade", "people_segment_code")`.
    """
    rows = []
    for seg, cuts in _GRADE_CUTS.items():
        edges = [None, *cuts, None]
        rows += [{"segment": seg, "lo": edges[g - 1], "hi": edges[g], "grade": g,
                  "cell_id": cell_id("business_grade", GRADE_VERSION, seg, g)} for g in range(1, 13)]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": grade + "_ladder",
        "columns": {"segment": "Int64", "lo": "Float64", "hi": "Float64", "grade": "Int64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": segment, "value_column": "segment"},
            {"type": "between", "variable": probability, "lower_bound_column": "lo",
             "upper_bound_column": "hi", "allow_gaps": True}]},
        "outputs": ["grade", "cell_id"],
        "default": [12, None],
    }).relabel(writes={"grade": grade, "cell_id": f"{grade}_cell_id"})


def segment_code(is_natural_person: bool) -> float:
    return float(SEGMENT_SME_PEOPLE if is_natural_person else SEGMENT_SME_ENTITY)


def entity_scoring_unit():
    """Score, overlay, calibration and grade for one entity; reads the entity's own fields."""
    overlay = ENTITY_OVERLAYS.apply_stack_step(
        "entity_score", ENTITY_ADJUSTMENT_SET_ID, base_field="score_before_overlay").relabel(
        writes={"adjustment_set_id": "score_adjustment_set_id", "adjustments_applied": "score_adjustments_applied"})
    pd = calibration.probability_of_default_step
    return flow(
        step(segment_code), build_entity_scorecard(), overlay, calibration.build_calibration_table(),
        pd.relabel(reads={"score": "entity_score"}, writes={"probability_of_default": "entity_pd"}),
        pd.named("unadjusted").relabel(reads={"score": "entity_score_unadjusted"},
                                       writes={"probability_of_default": "entity_pd_unadjusted"}),
        grade_lookup("entity_pd", "entity_grade"),
        grade_lookup("entity_pd_unadjusted", "entity_grade_unadjusted"),
        name="scoring",
    )
