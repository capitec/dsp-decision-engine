"""Score to probability of default to risk grade, with the probability overlay between the two."""
from __future__ import annotations

import math

from decider import DecisionTableConfig, step

from credit_core.evidence import cell_id
from retail_credit.documents import recorded

CALIBRATION_VERSION = "cal-2026.09"
RISK_GRADE_VERSION = "rg-2026.09"

# segment_code -> (anchor, scale) of the logistic curve
_CALIBRATION = {
    1: (560.0, 65.0), 2: (580.0, 62.0), 3: (590.0, 60.0), 4: (600.0, 60.0), 5: (590.0, 58.0), 6: (595.0, 63.0),
    7: (605.0, 60.0), 8: (600.0, 60.0), 9: (585.0, 64.0), 10: (595.0, 62.0), 11: (615.0, 58.0), 12: (600.0, 60.0),
}
GRADES = 12


def _grade_cut_points(segment: int) -> list[float]:
    """Eleven ascending probabilities separating twelve grades; the later segments are given wider grades."""
    return [0.008 + i * (0.55 - 0.008) / 11 * (1.0 + 0.05 * (segment - 4)) for i in range(GRADES - 1)]


def build_calibration_table() -> DecisionTableConfig:
    """Writes `anchor`, `scale` and `calibration_cell_id` for `segment_code`."""
    rows = [{"segment": segment, "anchor": anchor, "scale": scale,
             "cell_id": cell_id("calibration", CALIBRATION_VERSION, segment)}
            for segment, (anchor, scale) in _CALIBRATION.items()]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "calibration",
        "columns": {"segment": "Int64", "anchor": "Float64", "scale": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "eq", "variable": "segment_code", "value_column": "segment"},
        "outputs": ["anchor", "scale", "cell_id"], "default": [None, None, None],
    }).relabel(writes={"cell_id": "calibration_cell_id"})


def build_risk_grade_table() -> DecisionTableConfig:
    """Writes `risk_grade` (1 best) and `risk_grade_cell_id` for `segment_code` and `probability_of_default`."""
    rows = []
    for segment in _CALIBRATION:
        edges = [-math.inf, *_grade_cut_points(segment), math.inf]
        rows += [{"segment": segment, "lo": edges[grade - 1], "hi": edges[grade], "grade": grade,
                  "cell_id": cell_id("risk_grade", RISK_GRADE_VERSION, segment, grade)}
                 for grade in range(1, GRADES + 1)]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "risk_grade",
        "columns": {"segment": "Int64", "lo": "Float64", "hi": "Float64", "grade": "Int64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "segment_code", "value_column": "segment"},
            {"type": "between", "variable": "probability_of_default", "lower_bound_column": "lo",
             "upper_bound_column": "hi", "allow_gaps": True}]},
        "outputs": ["grade", "cell_id"], "default": [None, None],
    }).relabel(writes={"grade": "risk_grade", "cell_id": "risk_grade_cell_id"})


def probability_of_default_raw(score: float, anchor: float, scale: float) -> float:
    return 1.0 / (1.0 + math.exp((score - anchor) / scale))


def probability_of_default_overlay_step(pd_overlays):
    """The overlay runs between calibration and grading, so a tightening moves the grade."""
    return recorded(pd_overlays, "pd")


probability_of_default_raw_step = step(probability_of_default_raw)
