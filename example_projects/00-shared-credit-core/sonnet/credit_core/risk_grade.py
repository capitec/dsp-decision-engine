"""`core.risk_grade` -- grade assignment (spec 00 §6.12).

12 PD-band boundaries per product per segment (00 §8: ~8 x 5 x 12 in full;
this slice carries 2 products x 2 segments x 12 grades -- working depth,
same mechanism, smaller population). Grade 1 is best (lowest PD), grade 12
worst. The table's `outputs` carry the band edges that produced the grade,
because "which boundaries produced this grade" is itself required evidence
(00 §6.12 "Produces").
"""
from __future__ import annotations

from decider.steps.tables import DecisionTableConfig

from credit_core.evidence import cell_id as _cell_id

RISK_GRADE_VERSION = "rg-2026.09"

# (product_code, segment_code) -> 11 ascending PD cut points separating 12 grades.
_BOUNDARIES = {
    (10, 1): [0.01, 0.02, 0.035, 0.05, 0.07, 0.10, 0.14, 0.19, 0.26, 0.35, 0.50],
    (10, 2): [0.008, 0.016, 0.028, 0.04, 0.056, 0.08, 0.11, 0.15, 0.21, 0.29, 0.42],
    (10, 3): [0.015, 0.03, 0.05, 0.075, 0.105, 0.15, 0.21, 0.28, 0.37, 0.48, 0.62],  # new-to-bank: wider bands
    (20, 1): [0.012, 0.024, 0.04, 0.06, 0.085, 0.12, 0.165, 0.22, 0.30, 0.40, 0.55],
    (20, 2): [0.009, 0.018, 0.03, 0.045, 0.063, 0.09, 0.125, 0.17, 0.235, 0.32, 0.46],
}


def build_risk_grade_table() -> DecisionTableConfig:
    rows = []
    for (product, segment), cuts in _BOUNDARIES.items():
        # +-inf, not None: None-as-open-edge only works once per table (DecisionTableConfig
        # requires exactly one open lower and one open upper row overall), and this table
        # repeats the ladder once per (product, segment) group.
        edges = [float("-inf"), *cuts, float("inf")]
        for grade in range(1, 13):
            lo, hi = edges[grade - 1], edges[grade]
            rows.append({
                "product": product, "segment": segment, "lo": lo, "hi": hi, "grade": grade,
                "cell_id": _cell_id("risk_grade", RISK_GRADE_VERSION, product, segment, grade),
                "risk_grade_version": RISK_GRADE_VERSION,
            })
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "risk_grade_boundaries",
        "columns": {"product": "Int64", "segment": "Int64", "lo": "Float64", "hi": "Float64", "grade": "Int64",
                    "cell_id": "String", "risk_grade_version": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "product_code", "value_column": "product"},
            {"type": "eq", "variable": "segment_code", "value_column": "segment"},
            {"type": "between", "variable": "probability_of_default", "lower_bound_column": "lo",
             "upper_bound_column": "hi", "allow_gaps": True},
        ]},
        "outputs": ["grade", "lo", "hi", "cell_id", "risk_grade_version"],
        "default": [12, None, None, None, RISK_GRADE_VERSION],
    }).relabel(writes={"cell_id": "risk_grade_cell_id", "lo": "risk_grade_boundary_lo",
                        "hi": "risk_grade_boundary_hi"})


def risk_grade_output(grade: int) -> int:
    """Renames the table's `grade` output to the vocabulary name `risk_grade`."""
    return grade
