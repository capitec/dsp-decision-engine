"""Entity scoring: a scorecard, a tighten-only overlay, calibration to a probability of default, and a grade.

`entity_scoring_unit()` is the flow `each` runs on every entity, so each entity carries its own score,
probability of default and grade, beside the unadjusted ones. The grade ladder is also used by the people
and financial components, which grade a blended probability the same way.
"""
from __future__ import annotations

from types import SimpleNamespace

from decider import DecisionTableConfig, ScorecardConfig, flow, step

from credit_core import calibration
from credit_core.adjustments import AdjustmentStack
from credit_core.evidence import cell_id
from credit_core.vocab import SEGMENT_SME_ENTITY, SEGMENT_SME_PEOPLE

from business_nested.documents import shipped

GRADE_VERSION = "bus-rg-2026.09"

# segment -> the 11 ascending probability-of-default cut points between grades 1 (best) and 12.
_GRADE_CUTS = {
    SEGMENT_SME_PEOPLE: [0.01, 0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.25, 0.34, 0.45, 0.60],
    SEGMENT_SME_ENTITY: [0.008, 0.018, 0.032, 0.05, 0.075, 0.11, 0.155, 0.21, 0.29, 0.39, 0.52],
}

def build_entity_scorecard() -> ScorecardConfig:
    """Four characteristics: bureau score, months on record, worst delinquency and ownership stake.

    Writes `score_before_overlay`. The bureau cut-offs are params; a missing characteristic scores its default bin.
    """
    def edge(name, default):
        return {"param": name, "default": default}

    return ScorecardConfig.load({
        "type": "scorecard", "name": "entity_scorecard", "output_name": "score_before_overlay",
        "variables": [
            {"type": "constant", "score": 700, "output_name": "entity_scorecard_offset"},
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


def grade_lookup(probability: str, grade: str, key: str = "segment_code", cuts: dict | None = None,
                 version: str = GRADE_VERSION) -> DecisionTableConfig:
    """Grade 1 (best) to 12 for the probability of default in column `probability`.

    The ladder is picked by the column `key` (a segment: 4 people, 5 juristic); any other value gets grade 12.
    `cuts` maps each key value to its 11 ascending cut points. Writes `grade` and `<grade>_cell_id`.

    Example: `grade_lookup("people_pd", "people_grade", "people_segment_code")`.
    """
    rows = []
    for value, ladder in (cuts or _GRADE_CUTS).items():
        edges = [None, *ladder, None]
        rows += [{"key": value, "lo": edges[g - 1], "hi": edges[g], "grade": g,
                  "cell_id": cell_id("grade_ladder", version, value, g)} for g in range(1, 13)]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": grade + "_ladder",
        "columns": {"key": "Int64", "lo": "Float64", "hi": "Float64", "grade": "Int64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": key, "value_column": "key"},
            {"type": "between", "variable": probability, "lower_bound_column": "lo",
             "upper_bound_column": "hi", "allow_gaps": True}]},
        "outputs": ["grade", "cell_id"],
        "default": [12, None],
    }).relabel(writes={"grade": grade, "cell_id": f"{grade}_cell_id"})


def segment_code(is_natural_person: bool) -> float:
    return float(SEGMENT_SME_PEOPLE if is_natural_person else SEGMENT_SME_ENTITY)


def entity_scoring_unit(score_adjustments: AdjustmentStack | None = None):
    """Score, overlay, calibration and grade for one entity; reads the entity's own fields.

    `score_adjustments` is the `entity_score_adjustments` document; by default the one in `configs/0.2.0`.
    """
    overlay = (score_adjustments or shipped("entity_score_adjustments")).relabel(
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
    ).emit("score_before_overlay", "entity_score", "entity_score_unadjusted", "score_adjustments_applied", "entity_pd",
           "entity_pd_unadjusted", "entity_grade", "entity_grade_unadjusted")


# governance-replay lists the overlays through this name; no step reads it.
ENTITY_OVERLAYS = SimpleNamespace(_all=shipped("entity_score_adjustments").adjustments)
