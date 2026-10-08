"""Risk grade 1 (best) to 12 (worst) from the probability of default, per product and segment."""
from __future__ import annotations

from decider import DecisionTableConfig

from credit_core.evidence import cell_id

RISK_GRADE_VERSION = "rg-2026.09"

# (product_code, segment_code) -> the 11 ascending PD cut points between 12 grades.
_BOUNDARIES = {
    (10, 1): [0.01, 0.02, 0.035, 0.05, 0.07, 0.10, 0.14, 0.19, 0.26, 0.35, 0.50],
    (10, 2): [0.008, 0.016, 0.028, 0.04, 0.056, 0.08, 0.11, 0.15, 0.21, 0.29, 0.42],
    (10, 3): [0.015, 0.03, 0.05, 0.075, 0.105, 0.15, 0.21, 0.28, 0.37, 0.48, 0.62],  # new to bank: wider bands
    (20, 1): [0.012, 0.024, 0.04, 0.06, 0.085, 0.12, 0.165, 0.22, 0.30, 0.40, 0.55],
    (20, 2): [0.009, 0.018, 0.03, 0.045, 0.063, 0.09, 0.125, 0.17, 0.235, 0.32, 0.46],
}


def build_risk_grade_table() -> DecisionTableConfig:
    """Writes `risk_grade`, the band edges that produced it (`risk_grade_boundary_lo`/`_hi`), the cell and version.

    A product and segment with no boundaries gets grade 12.
    """
    rows = []
    for (product, segment), cuts in _BOUNDARIES.items():
        # Each (product, segment) is its own ladder: the first band has no lower edge, the last no upper one.
        edges = [None, *cuts, None]
        for grade in range(1, 13):
            rows.append({"product": product, "segment": segment, "lo": edges[grade - 1], "hi": edges[grade],
                         "grade": grade, "cell_id": cell_id("risk_grade", RISK_GRADE_VERSION, product, segment, grade),
                         "risk_grade_version": RISK_GRADE_VERSION})
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "risk_grade_boundaries",
        "columns": {"product": "Int64", "segment": "Int64", "lo": "Float64", "hi": "Float64", "grade": "Int64",
                    "cell_id": "String", "risk_grade_version": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "product_code", "value_column": "product"},
            {"type": "eq", "variable": "segment_code", "value_column": "segment"},
            {"type": "between", "variable": "probability_of_default", "lower_bound_column": "lo",
             "upper_bound_column": "hi"},
        ]},
        "outputs": ["grade", "lo", "hi", "cell_id", "risk_grade_version"],
        "default": [12, None, None, None, RISK_GRADE_VERSION],
    }).relabel(writes={"grade": "risk_grade", "cell_id": "risk_grade_cell_id", "lo": "risk_grade_boundary_lo",
                       "hi": "risk_grade_boundary_hi"})
