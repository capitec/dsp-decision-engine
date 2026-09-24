"""§5.3 -- behavioural scoring: `behaviour_score`, `probability_of_default`,
`behaviour_grade`. Project-specific (README §4 Q4: "The behavioural
scorecard... [is] project-specific"), built entirely from `decider`
built-ins (`ScorecardConfig`, `DecisionTableConfig`) and `core.adjustments`
-- the same mechanism 00's `core.scorecard`/`core.calibration`/
`core.risk_grade` use, not reused code, because this scorecard evaluates
different characteristics (behaviour, not application) and produces a
*distinct* grade from `risk_grade` (§4.3: "behaviour_grade... Distinct from
risk_grade").

Working depth (SCOPE.md "cut breadth"): 8 characteristics per product
(one family representative each), not the spec's 32, mirroring 00's own
"8, not 20-60" precedent in `credit_core/scorecard.py`.
"""
from __future__ import annotations

import math

from decider import step
from decider.steps.scorecard import ScorecardConfig
from decider.steps.tables import DecisionTableConfig

from credit_core.evidence import cell_id as _cell_id

SCORECARD_VERSION = "behaviour-sc-2026.09"
CALIBRATION_VERSION = "behaviour-cal-2026.09"
GRADE_VERSION = "behaviour-grade-2026.09"


def build_behaviour_scorecard() -> ScorecardConfig:
    """One card, keyed by `product_code` through per-variable params rather than two
    separate configs -- "one scorecard per product" (§5.3) expressed as one shape with
    product-specific bin edges, exactly `core.calibration`'s segment-table precedent."""
    return ScorecardConfig.load({
        "type": "scorecard", "name": "behaviour_scorecard",
        "output_name": "behaviour_score_raw",
        "variables": [
            {"type": "constant", "score": 600, "output_name": "base_score"},
            {"type": "scored", "variable_name": "worst_arrears_months_24", "strict": False,
             "default": {"value": 0, "name": "insufficient_history"}, "bins": [
                 {"value": 30, "upper_bound": 0.5, "name": "never"},
                 {"value": 5, "lower_bound": 0.5, "upper_bound": 2.0, "name": "minor"},
                 {"value": -45, "lower_bound": 2.0, "name": "material"},
             ]},
            {"type": "scored", "variable_name": "months_since_last_arrears", "strict": False,
             "default": {"value": 20, "name": "no_arrears_on_record"}, "bins": [
                 {"value": -30, "upper_bound": 6.0, "name": "recent"},
                 {"value": -5, "lower_bound": 6.0, "upper_bound": 18.0, "name": "moderate"},
                 {"value": 15, "lower_bound": 18.0, "name": "distant"},
             ]},
            {"type": "scored", "variable_name": "payment_to_balance_ratio_6m", "strict": False,
             "default": {"value": -5, "name": "insufficient_history"}, "bins": [
                 {"value": -25, "upper_bound": 0.15, "name": "minimum_only"},
                 {"value": 5, "lower_bound": 0.15, "upper_bound": 0.5, "name": "moderate"},
                 {"value": 25, "lower_bound": 0.5, "name": "strong"},
             ]},
            {"type": "scored", "variable_name": "mean_utilisation_6m", "strict": False,
             "default": {"value": 0, "name": "no_revolving"}, "bins": [
                 {"value": 15, "upper_bound": 0.30, "name": "low"},
                 {"value": 0, "lower_bound": 0.30, "upper_bound": 0.70, "name": "mid"},
                 {"value": -15, "lower_bound": 0.70, "name": "high"},
             ]},
            {"type": "scored", "variable_name": "over_limit_cycles_12m", "strict": False,
             "default": {"value": 10, "name": "never_over_limit"}, "bins": [
                 {"value": 10, "items": [0]},
                 {"value": -10, "items": [1, 2]},
                 {"value": -30, "items": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
             ]},
            {"type": "scored", "variable_name": "cash_withdrawal_ratio_12m", "strict": False,
             "default": {"value": 0, "name": "no_cash_use"}, "bins": [
                 {"value": 5, "upper_bound": 0.15, "name": "low"},
                 {"value": -10, "lower_bound": 0.15, "upper_bound": 0.45, "name": "moderate"},
                 {"value": -25, "lower_bound": 0.45, "name": "high"},
             ]},
            {"type": "scored", "variable_name": "months_on_book", "strict": False,
             "default": {"value": 0, "name": "unknown"}, "bins": [
                 {"value": -10, "upper_bound": 12.0, "name": "under_1y"},
                 {"value": 5, "lower_bound": 12.0, "upper_bound": 60.0, "name": "1_to_5y"},
                 {"value": 15, "lower_bound": 60.0, "name": "over_5y"},
             ]},
            {"type": "scored", "variable_name": "bureau_enquiry_velocity_6m", "strict": False,
             "default": {"value": -10, "name": "no_bureau_record"}, "bins": [
                 {"value": 10, "upper_bound": 2.5, "name": "low"},
                 {"value": -5, "lower_bound": 2.5, "upper_bound": 6.0, "name": "moderate"},
                 {"value": -20, "lower_bound": 6.0, "name": "high"},
             ]},
        ],
    })


_PD_ANCHOR_SCALE = {20: (600.0, 65.0), 21: (580.0, 60.0)}  # product_code -> (anchor, scale)


def build_pd_calibration_table() -> DecisionTableConfig:
    rows = [
        {"product": product, "anchor": anchor, "scale": scale,
         "cell_id": _cell_id("behaviour_calibration", CALIBRATION_VERSION, product)}
        for product, (anchor, scale) in _PD_ANCHOR_SCALE.items()
    ]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "behaviour_pd_calibration",
        "columns": {"product": "Int64", "anchor": "Float64", "scale": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "eq", "variable": "product_code", "value_column": "product"},
        "outputs": ["anchor", "scale", "cell_id"],
        "default": [600.0, 65.0, None],
    }).relabel(writes={"cell_id": "behaviour_calibration_cell_id"})


def probability_of_default(behaviour_score: float, anchor: float, scale: float) -> float:
    return round(1.0 / (1.0 + math.exp((behaviour_score - anchor) / scale)), 6)


probability_of_default_step = step(probability_of_default)


# behaviour_grade (§4.3): 1 best .. 12 worst, from PD boundaries, per product (§6.1: "2
# products x 12 grade edges"). Widening bands at the tails, same shape as 00's risk_grade.
_GRADE_BOUNDARIES = {
    20: [0.01, 0.02, 0.035, 0.05, 0.07, 0.10, 0.14, 0.19, 0.26, 0.35, 0.50],
    21: [0.015, 0.028, 0.045, 0.065, 0.09, 0.125, 0.17, 0.23, 0.31, 0.41, 0.55],
}


def build_behaviour_grade_table() -> DecisionTableConfig:
    rows = []
    for product, cuts in _GRADE_BOUNDARIES.items():
        edges = [float("-inf"), *cuts, float("inf")]
        for grade in range(1, 13):
            lo, hi = edges[grade - 1], edges[grade]
            rows.append({
                "product": product, "lo": lo, "hi": hi, "grade": grade,
                "cell_id": _cell_id("behaviour_grade", GRADE_VERSION, product, grade),
            })
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "behaviour_grade_boundaries",
        "columns": {"product": "Int64", "lo": "Float64", "hi": "Float64", "grade": "Int64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "product_code", "value_column": "product"},
            {"type": "between", "variable": "probability_of_default", "lower_bound_column": "lo",
             "upper_bound_column": "hi", "allow_gaps": True},
        ]},
        "outputs": ["grade", "cell_id"],
        "default": [12, None],
    }).relabel(writes={"grade": "behaviour_grade", "cell_id": "behaviour_grade_cell_id"})
