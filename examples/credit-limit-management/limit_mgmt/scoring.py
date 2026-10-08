"""Behavioural scoring: a points scorecard, a score-to-PD curve per product and a PD-to-grade ladder.

Tighten-only overlays (a score shift, a PD multiplier) sit between the stages and leave the unadjusted values beside
the adjusted ones.
"""
from __future__ import annotations

from decider import DecisionTableConfig, ScorecardConfig, flow

from credit_core import calibration
from credit_core.evidence import cell_id
from limit_mgmt.overlays import BEHAVIOUR_ADJUSTMENTS, SET_ID

SCORECARD_VERSION = "behaviour-sc-2026.09"
CALIBRATION_VERSION = "behaviour-cal-2026.09"
GRADE_VERSION = "behaviour-grade-2026.09"


def _scored(variable: str, default: tuple[int, str], *bins: tuple) -> dict:
    """Bins are `(points, name, lower, upper)`; a bound of `None` is open."""
    return {"type": "scored", "variable_name": variable, "strict": False,
            "default": {"value": default[0], "name": default[1]},
            "bins": [{"value": v, "name": n, **({"lower_bound": lo} if lo is not None else {}),
                      **({"upper_bound": hi} if hi is not None else {})} for v, n, lo, hi in bins]}


def build_behaviour_scorecard() -> ScorecardConfig:
    """Eight characteristics on a base of 600; scores run from about 400 to 725. Writes `behaviour_score_raw`."""
    return ScorecardConfig.load({
        "type": "scorecard", "name": "behaviour_scorecard", "output_name": "behaviour_score_raw",
        "variables": [
            {"type": "constant", "score": 600, "output_name": "base_score"},
            _scored("worst_arrears_months_24", (0, "insufficient_history"),
                    (30, "never", None, 0.5), (5, "minor", 0.5, 2.0), (-45, "material", 2.0, None)),
            _scored("months_since_last_arrears", (20, "no_arrears_on_record"),
                    (-30, "recent", None, 6.0), (-5, "moderate", 6.0, 18.0), (15, "distant", 18.0, None)),
            _scored("payment_to_balance_ratio_6m", (-5, "insufficient_history"),
                    (-25, "minimum_only", None, 0.15), (5, "moderate", 0.15, 0.5), (25, "strong", 0.5, None)),
            _scored("revolving_utilisation_6m", (0, "no_revolving"),
                    (15, "low", None, 0.30), (0, "mid", 0.30, 0.70), (-15, "high", 0.70, None)),
            {"type": "scored", "variable_name": "over_limit_cycles_12m", "strict": False,
             "default": {"value": 10, "name": "never_over_limit"}, "bins": [
                 {"value": 10, "items": [0]}, {"value": -10, "items": [1, 2]},
                 {"value": -30, "items": list(range(3, 13))}]},
            _scored("cash_withdrawal_ratio_12m", (0, "no_cash_use"),
                    (5, "low", None, 0.15), (-10, "moderate", 0.15, 0.45), (-25, "high", 0.45, None)),
            _scored("months_on_book", (0, "unknown"),
                    (-10, "under_1y", None, 12.0), (5, "1_to_5y", 12.0, 60.0), (15, "over_5y", 60.0, None)),
            _scored("bureau_enquiry_velocity_6m", (-10, "no_bureau_record"),
                    (10, "low", None, 2.5), (-5, "moderate", 2.5, 6.0), (-20, "high", 6.0, None)),
        ],
    })


# product -> (anchor score, scale): PD is 50% at the anchor. The Access Facility book runs riskier.
_CURVES = {20: (490.0, 52.0), 21: (470.0, 50.0)}

# product -> the 11 ascending PD cut points between grades 1 (best) and 12.
_GRADE_CUTS = {
    20: [0.01, 0.02, 0.035, 0.05, 0.07, 0.10, 0.14, 0.19, 0.26, 0.35, 0.50],
    21: [0.015, 0.028, 0.045, 0.065, 0.09, 0.125, 0.17, 0.23, 0.31, 0.41, 0.55],
}


def build_pd_calibration_table() -> DecisionTableConfig:
    rows = [{"product": product, "anchor": anchor, "scale": scale,
             "cell_id": cell_id("behaviour_calibration", CALIBRATION_VERSION, product)}
            for product, (anchor, scale) in _CURVES.items()]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "behaviour_pd_calibration",
        "columns": {"product": "Int64", "anchor": "Float64", "scale": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "eq", "variable": "product_code", "value_column": "product"},
        "outputs": ["anchor", "scale", "cell_id"],
        "default": [*_CURVES[20], None],
    }).relabel(writes={"cell_id": "behaviour_calibration_cell_id"})


def build_behaviour_grade_table() -> DecisionTableConfig:
    rows = []
    for product, cuts in _GRADE_CUTS.items():
        edges = [None, *cuts, None]  # open at both ends, once per product
        rows += [{"product": product, "lo": edges[g - 1], "hi": edges[g], "grade": g,
                  "cell_id": cell_id("behaviour_grade", GRADE_VERSION, product, g)} for g in range(1, 13)]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "behaviour_grade_boundaries",
        "columns": {"product": "Int64", "lo": "Float64", "hi": "Float64", "grade": "Int64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "product_code", "value_column": "product"},
            {"type": "between", "variable": "probability_of_default", "lower_bound_column": "lo",
             "upper_bound_column": "hi"}]},
        "outputs": ["grade", "cell_id"],
        "default": [12, None],
    }).relabel(writes={"grade": "behaviour_grade", "cell_id": "behaviour_grade_cell_id"})


def unit():
    score_adjustment = BEHAVIOUR_ADJUSTMENTS.apply_stack_step(
        "behaviour_score", SET_ID, base_field="behaviour_score_raw", unadjusted_output="behaviour_score_unadjusted",
    ).relabel(writes={"adjustment_set_id": "score_adjustment_set_id", "adjustments_applied": "score_adjustments_applied"})
    pd_adjustment = BEHAVIOUR_ADJUSTMENTS.apply_stack_step(
        "probability_of_default", SET_ID, base_field="probability_of_default_raw",
        unadjusted_output="probability_of_default_unadjusted",
    ).relabel(writes={"adjustment_set_id": "pd_adjustment_set_id", "adjustments_applied": "pd_adjustments_applied"})
    pd_raw = calibration.probability_of_default_step.relabel(
        reads={"score": "behaviour_score"}, writes={"probability_of_default": "probability_of_default_raw"})
    return flow(build_behaviour_scorecard(), score_adjustment, build_pd_calibration_table(), pd_raw, pd_adjustment,
                build_behaviour_grade_table(), name="scoring")
