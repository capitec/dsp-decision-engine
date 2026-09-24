"""P08 -- Calibration, grading and adjustments (spec 10 §5.9): this project's own tables.

Three things, in the spec's fixed order (O-01: adjustments before grading):
(a) calibration -- score to PD, invertible by construction, reusing
`core.calibration`'s exact-inverse logistic shape but with product 10's
own anchor/scale; (b) the overlay stack, resolved **once** for the whole
decision (O-05, O-21) via `core.adjustments.AdjustmentRegister` -- reused
directly, not reimplemented, because the register *is* the capability;
(c) grading -- PD to `risk_grade` 1..12, product 10's own boundary table.

Reused from 00: `AdjustmentRegister`/`Adjustment`/`AdjustmentEffect` (the
mechanism), `DecisionTableConfig` (the table shape). Written here: this
project's own calibration curve and grade boundaries, and its own overlay
register (10 §5.9's worked example: ADJ-0061, a channel-3 odds multiplier).
"""
from __future__ import annotations

import math

from decider import step
from decider.steps.tables import DecisionTableConfig

from credit_core.evidence import cell_id as _cell_id
from retail_credit.overlays import ADJUSTMENT_SET_ID, OVERLAY_REGISTER  # noqa: F401 -- re-exported for pipeline.py

CALIBRATION_VERSION = "p10-cal-2026.09"
RISK_GRADE_VERSION = "p10-rg-2026.09"

# segment_code -> (anchor, scale), product 10 only.
_CALIBRATION_SEGMENTS = {
    1: (560.0, 65.0), 2: (580.0, 62.0), 3: (590.0, 60.0), 4: (600.0, 60.0),
    5: (590.0, 58.0), 6: (595.0, 63.0), 7: (605.0, 60.0), 8: (600.0, 60.0),
    9: (585.0, 64.0), 10: (595.0, 62.0), 11: (615.0, 58.0), 12: (600.0, 60.0),
}

# 11 ascending PD cut points separating 12 grades, per segment -- product 10 only
# (10 §5.9(c): "12 x 12 x 12" in full; working depth here, same mechanism).
_BOUNDARIES = {
    seg: [0.008 + i * (0.55 - 0.008) / 11 * (1.0 + 0.05 * (seg - 4)) for i in range(11)]
    for seg in range(1, 13)
}


def build_calibration_table() -> DecisionTableConfig:
    rows = [
        {"segment": seg, "anchor": anchor, "scale": scale,
         "cell_id": _cell_id("p10.calibration", CALIBRATION_VERSION, seg),
         "calibration_version": CALIBRATION_VERSION}
        for seg, (anchor, scale) in _CALIBRATION_SEGMENTS.items()
    ]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "p10_calibration_segments",
        "columns": {"segment": "Int64", "anchor": "Float64", "scale": "Float64", "cell_id": "String",
                    "calibration_version": "String"},
        "rows": rows,
        "expression": {"type": "eq", "variable": "segment_code", "value_column": "segment"},
        "outputs": ["anchor", "scale", "cell_id", "calibration_version"],
        "default": [None, None, None, None],
    })


def probability_of_default(score: float, anchor: float, scale: float) -> float:
    return 1.0 / (1.0 + math.exp((score - anchor) / scale))


def score_for_probability(probability_of_default: float, anchor: float, scale: float) -> float:
    """The exact inverse (10 §5.9(a): "the two directions must agree to 1e-6")."""
    p = min(max(probability_of_default, 1e-9), 1 - 1e-9)
    return anchor - scale * math.log(p / (1 - p))


probability_of_default_step = step(probability_of_default, output="probability_of_default_raw")
score_for_probability_step = step(score_for_probability)


def build_risk_grade_table() -> DecisionTableConfig:
    rows = []
    for seg, cuts in _BOUNDARIES.items():
        edges = [float("-inf"), *cuts, float("inf")]
        for grade in range(1, 13):
            lo, hi = edges[grade - 1], edges[grade]
            rows.append({
                "segment": seg, "lo": lo, "hi": hi, "grade": grade,
                "cell_id": _cell_id("p10.risk_grade", RISK_GRADE_VERSION, seg, grade),
                "risk_grade_version": RISK_GRADE_VERSION, "boundary_lo": lo, "boundary_hi": hi,
            })
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "p10_risk_grade",
        "columns": {"segment": "Int64", "lo": "Float64", "hi": "Float64", "grade": "Int64",
                    "cell_id": "String", "risk_grade_version": "String",
                    "boundary_lo": "Float64", "boundary_hi": "Float64"},
        "rows": rows,
        "expression": {
            "type": "and",
            "expressions": [
                {"type": "eq", "variable": "segment_code", "value_column": "segment"},
                {"type": "between", "variable": "probability_of_default", "lower_bound_column": "lo",
                 "upper_bound_column": "hi", "allow_gaps": True},
            ],
        },
        "outputs": ["grade", "cell_id", "risk_grade_version", "boundary_lo", "boundary_hi"],
        "default": [None, None, None, None, None],
    })


def risk_grade_output(grade: int) -> int:
    return grade


risk_grade_output_step = step(risk_grade_output, output="risk_grade")


def probability_of_default_adjustment_step():
    """P08 resolves the overlay register once, for `probability_of_default` (O-01: before
    grading; O-05/O-21: this is the one and only resolution for the whole decision).
    """
    return OVERLAY_REGISTER.apply_stack_step(
        "probability_of_default", ADJUSTMENT_SET_ID, base_field="probability_of_default_raw",
        unadjusted_output="probability_of_default_unadjusted",
    )
