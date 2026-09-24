"""Stage 5 -- entity scoring (spec 05 §5.7), one scorecard family (SCOPE.md).

The natural-person/juristic split into two scorecard families (`BUS-PERS-01`
/ `BUS-COMM-01`) and the thin-file/no-hit/no-enquiry-possible situation
split are both out of this slice (SCOPE.md: "entity scoring for one
family"); every entity runs through one small scorecard. What *is* kept at
full depth is the reuse shape spec 05 §2 names as its sharpest form of the
reuse question: `core.scorecard`'s mechanism (a fresh `ScorecardConfig`, the
one project-specific part), `core.calibration` (project 00's, unmodified,
keyed by `segment_code` -- natural persons calibrate on
`SEGMENT_SME_PEOPLE`, juristic entities on `SEGMENT_SME_ENTITY`, both
already present in 00's calibration table, "same capability twice with
different settings" per entity rather than per application) and
`core.adjustments` (project 00's `AdjustmentRegister.apply_stack_step`,
scoped on `segment_code` -- one of its four built-in provenance keys, so no
custom overlay wiring was needed here, unlike `events.py`'s threshold
overlay whose scope (`criticality_class`) isn't one of the four).

**Framework note (spec 05 §13 Q1, Q9).** There is no `decider` construct
that runs a step, or a small pipeline of them, once per element of a
ragged collection nested inside a row. The entity-level sub-pipeline below
(scorecard -> overlay -> calibration -> risk grade) is built and bound
**once**, at import time, exactly as `pipeline.py` binds the outer one; the
only difference is *where* it runs -- inside this stage's `frame_step`,
against a small per-application `entities` frame built from the row's own
flat entity arrays, through a second `Engine.run()` call. This composes
`ScorecardConfig`/`DecisionTableConfig` genuinely (the real objects, the
real engine), not a hand-rolled re-implementation of bin-to-points scoring
-- confirmed to work in both `.score()` and `.run()` (batch) modes with
zero and many entities. See NOTES.md "Framework friction" for why this is
a workaround rather than a documented pattern.
"""
from __future__ import annotations

from datetime import date

import polars as pl

from decider import Engine, dag, frame_step, missing_as, param, step
from decider.steps.scorecard import ScorecardConfig
from decider.steps.tables import DecisionTableConfig

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.calibration import build_calibration_table, probability_of_default_step
from credit_core.evidence import cell_id as _cell_id

from business_nested import vocab

SCORECARD_ID = 2
SCORECARD_VERSION = "bus-sc-2026.09"
RISK_GRADE_VERSION = "bus-rg-2026.09"

# segment_code -> 11 ascending PD cut points separating 12 grades (spec 05 §5.7's own
# scorecard families each have their own boundaries in the real spec; one set here,
# reused across both segments -- SCOPE's "one family" extends to one boundary set too).
_BOUNDARIES = {
    vocab.SEGMENT_SME_PEOPLE: [0.01, 0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.25, 0.34, 0.45, 0.60],
    vocab.SEGMENT_SME_ENTITY: [0.008, 0.018, 0.032, 0.05, 0.075, 0.11, 0.155, 0.21, 0.29, 0.39, 0.52],
}


def build_entity_scorecard() -> ScorecardConfig:
    """Trade payment index, months on record, worst delinquency, ownership stake --
    four characteristics standing in for `BUS-COMM-01`/`BUS-PERS-01`'s larger sets
    (spec 05 §5.7 names 29/38); the mechanism is what this slice proves."""
    return ScorecardConfig.load({
        "type": "scorecard", "name": "entity_scorecard", "output_name": "entity_score_raw",
        "variables": [
            # Named `entity_scorecard_offset`, not `entity_base_score`: decider's typo
            # guard (`decider/registry/resolve.py` TYPO_CUTOFF=0.8, a difflib ratio over
            # names already produced earlier in the same scope) flags `entity_base_score`
            # against `entity_bureau_score` (ratio 0.833) as a likely typo and refuses to
            # bind it as a new input column at all -- see NOTES.md "Framework friction".
            {"type": "constant", "score": 600, "output_name": "entity_scorecard_offset"},
            {"type": "scored", "variable_name": "entity_bureau_score", "strict": False,
             "default": {"value": -20, "name": "no_bureau"}, "bins": [
                 {"value": -35, "upper_bound": {"param": "bus_bureau_low", "default": 500.0}, "name": "very_low"},
                 {"value": -5, "lower_bound": {"param": "bus_bureau_low", "default": 500.0},
                  "upper_bound": {"param": "bus_bureau_mid", "default": 650.0}, "name": "low"},
                 {"value": 20, "lower_bound": {"param": "bus_bureau_mid", "default": 650.0},
                  "upper_bound": {"param": "bus_bureau_high", "default": 750.0}, "name": "good"},
                 {"value": 40, "lower_bound": {"param": "bus_bureau_high", "default": 750.0}, "name": "excellent"},
             ]},
            {"type": "scored", "variable_name": "entity_months_on_record", "strict": False,
             "default": {"value": -10, "name": "unknown"}, "bins": [
                 {"value": -20, "upper_bound": 12.0, "name": "under_1y"},
                 {"value": 0, "lower_bound": 12.0, "upper_bound": 60.0, "name": "1_to_5y"},
                 {"value": 20, "lower_bound": 60.0, "name": "over_5y"},
             ]},
            {"type": "scored", "variable_name": "entity_worst_delinquency_months", "strict": False,
             "default": {"value": 0, "name": "no_history"}, "bins": [
                 {"value": 25, "upper_bound": 0.5, "name": "never"},
                 {"value": -5, "lower_bound": 0.5, "upper_bound": 3.0, "name": "minor"},
                 {"value": -30, "lower_bound": 3.0, "name": "material"},
             ]},
            {"type": "scored", "variable_name": "entity_effective_ownership_pct", "strict": False,
             "default": {"value": 0, "name": "unknown"}, "bins": [
                 # A larger stake means more skin in the game -- modest positive weight.
                 {"value": -5, "upper_bound": 10.0, "name": "small_stake"},
                 {"value": 0, "lower_bound": 10.0, "upper_bound": 50.0, "name": "mid_stake"},
                 {"value": 10, "lower_bound": 50.0, "name": "controlling_stake"},
             ]},
        ],
    })


def build_entity_risk_grade_table() -> DecisionTableConfig:
    rows = []
    for segment, cuts in _BOUNDARIES.items():
        edges = [float("-inf"), *cuts, float("inf")]
        for grade in range(1, 13):
            rows.append({
                "segment": segment, "lo": edges[grade - 1], "hi": edges[grade], "grade": grade,
                "cell_id": _cell_id("entity_risk_grade", RISK_GRADE_VERSION, segment, grade),
            })
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "entity_risk_grade_boundaries",
        "columns": {"segment": "Int64", "lo": "Float64", "hi": "Float64", "grade": "Int64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "segment_code", "value_column": "segment"},
            {"type": "between", "variable": "probability_of_default", "lower_bound_column": "lo",
             "upper_bound_column": "hi", "allow_gaps": True},
        ]},
        "outputs": ["grade", "cell_id"],
        "default": [12, None],
    }).relabel(writes={"cell_id": "entity_risk_grade_cell_id"})


def entity_segment_code(is_natural_person: bool) -> int:
    return vocab.SEGMENT_SME_PEOPLE if is_natural_person else vocab.SEGMENT_SME_ENTITY


def entity_risk_grade_output(grade: int) -> int:
    return grade


def grade_from_pd(probability_of_default: float, segment_code: int) -> int:
    """The same boundary ladder `build_entity_risk_grade_table()` loads into a
    `DecisionTableConfig`, as a plain function -- used by `blend.py` and `grade.py` to
    grade a blended PD without a second nested-`Engine` round trip for one lookup."""
    cuts = _BOUNDARIES[segment_code]
    grade = 1
    for cut in cuts:
        if probability_of_default < cut:
            return grade
        grade += 1
    return grade


def build_entity_overlays() -> tuple[AdjustmentRegister, str]:
    """A −12-point new-to-bank-segment overlay, tighten-only, on the natural-person
    entity segment -- the "reduced overlay stack" SCOPE.md asks for at entity level."""
    register = AdjustmentRegister([
        Adjustment(
            adjustment_id="ADJ-05-ENT-001", kind="score_shift", target="entity_score",
            effect=AdjustmentEffect("add", -12.0), scope={"segment_code": vocab.SEGMENT_SME_PEOPLE},
            stack_position=2, owner="Business Credit Risk Policy", approval_reference="CRC-2026-041",
            rationale="Natural-person entities behind new-to-bank businesses defaulting above prediction",
            effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31),
            tighten_only=True,
        ),
    ])
    return register, "AS-05-ENT-2026.09"


_ENTITY_OVERLAYS, _ENTITY_ADJUSTMENT_SET_ID = build_entity_overlays()


def _build_entity_pipeline():
    segment_step = step(entity_segment_code, output="segment_code").relabel(
        reads={"is_natural_person": "entity_is_natural_person"})
    scorecard_step = build_entity_scorecard().relabel(writes={"entity_score_raw": "entity_score_before_overlay"})
    overlay_step = _ENTITY_OVERLAYS.apply_stack_step(
        "entity_score", _ENTITY_ADJUSTMENT_SET_ID, base_field="entity_score_before_overlay",
    )
    pd_step = probability_of_default_step.relabel(
        reads={"score": "entity_score"}, writes={"probability_of_default": "entity_pd"},
    )
    pd_unadjusted_step = probability_of_default_step.relabel(
        reads={"score": "entity_score_unadjusted"}, writes={"probability_of_default": "entity_pd_unadjusted"},
    ).named("pd_unadjusted")
    grade_output_step = step(entity_risk_grade_output, output="entity_grade")

    return dag(
        segment_step, scorecard_step, overlay_step,
        build_calibration_table(),
        pd_step,
        pd_unadjusted_step,
        build_entity_risk_grade_table().relabel(reads={"probability_of_default": "entity_pd"}),
        grade_output_step,
        name="entity_scoring",
    ).emit(
        "entity_score_before_overlay", "entity_score", "entity_score_unadjusted", "adjustment_set_id",
        "adjustments_applied", "entity_pd", "entity_pd_unadjusted", "entity_grade", "entity_risk_grade_cell_id",
        "calibration_cell_id",
    )


_ENTITY_ENGINE = Engine().bind(_build_entity_pipeline())


def score_entities(
    entity_id: list[int], entity_is_natural_person: list[bool], entity_bureau_score: list,
    entity_months_on_record: list, entity_worst_delinquency_months: list, entity_effective_ownership_pct: list,
    decision_date,
) -> dict:
    if not entity_id:
        return {k: [] for k in (
            "entity_score", "entity_score_unadjusted", "entity_pd", "entity_pd_unadjusted", "entity_grade",
            "entity_adjustments_applied", "entity_scoring_adjustment_set_id",
        )}
    frame = pl.DataFrame({
        "entity_is_natural_person": entity_is_natural_person, "entity_bureau_score": entity_bureau_score,
        "entity_months_on_record": entity_months_on_record,
        "entity_worst_delinquency_months": entity_worst_delinquency_months,
        "entity_effective_ownership_pct": entity_effective_ownership_pct,
        "decision_date": [decision_date] * len(entity_id),
    })
    result = _ENTITY_ENGINE.run(frame, {})
    return {
        "entity_score": result["entity_score"].to_list(),
        "entity_score_unadjusted": result["entity_score_unadjusted"].to_list(),
        "entity_pd": result["entity_pd"].to_list(),
        "entity_pd_unadjusted": result["entity_pd_unadjusted"].to_list(),
        "entity_grade": result["entity_grade"].to_list(),
        # A flat, parallel per-entity list can't hold a per-entity *list* of adjustment ids
        # (that would be `list[list[str]]`, the same materialisation crash structure.py's
        # docstring documents) -- "|"-joined into one string per entity instead, exactly
        # `events.py`'s `ev_overlay_ids` convention.
        "entity_adjustments_applied": ["|".join(a) for a in result["adjustments_applied"].to_list()],
        "entity_scoring_adjustment_set_id": result["adjustment_set_id"].to_list(),
    }


_SCORE_OUTPUTS = ["entity_score", "entity_score_unadjusted", "entity_pd", "entity_pd_unadjusted", "entity_grade",
                  "entity_adjustments_applied", "entity_scoring_adjustment_set_id"]


@frame_step(
    reads=["entity_id", "entity_is_natural_person", "entity_bureau_score", "entity_months_on_record",
           "entity_worst_delinquency_months", "entity_effective_ownership_pct", "decision_date"],
    writes=_SCORE_OUTPUTS,
)
def score_entities_step(df: pl.DataFrame) -> pl.DataFrame:
    cols = ["entity_id", "entity_is_natural_person", "entity_bureau_score", "entity_months_on_record",
            "entity_worst_delinquency_months", "entity_effective_ownership_pct", "decision_date"]
    results = [score_entities(**{c: row[c] for c in cols}) for row in df.select(cols).to_dicts()]
    return df.with_columns(pl.DataFrame(results))
