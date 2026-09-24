"""`core.appetite` -- risk appetite limits (spec 00 §6.17; addendum A13: sector and facility-type dims).

Thin, frozen interface: a grade x product x segment grid (00 §8: ~12 x 8 x
6 in full; working depth here) returning the four limits and which one
binds. `sector_code` and `facility_type_code` are accepted but only
`None`-checked, not yet keyed into the grid -- declared so 05/11 can widen
this table without changing the interface (addendum A13).
"""
from __future__ import annotations

from decider import missing_as, param, step
from decider.steps.tables import DecisionTableConfig

from credit_core.evidence import cell_id as _cell_id

APPETITE_VERSION = "appetite-2026.09"

MAX_AMOUNT = "max_amount"
MAX_TERM = "max_term"
MAX_INSTALMENT_RATIO = "max_instalment_to_income_ratio"
MIN_PRICE = "min_price"

# (product_code, segment_code) -> per-grade (max_amount, max_term, max_ratio, min_price), grades 1..12.
_GRID = {
    (10, 1): lambda g: (500_000 - g * 35_000, 84, max(0.05, 0.45 - g * 0.03), 5.0 + g * 1.5),
    (10, 2): lambda g: (450_000 - g * 32_000, 72, max(0.05, 0.40 - g * 0.03), 5.5 + g * 1.5),
}


def build_appetite_table() -> DecisionTableConfig:
    rows = []
    for (product, segment), fn in _GRID.items():
        for grade in range(1, 13):
            max_amount, max_term, max_ratio, min_price = fn(grade)
            rows.append({
                "product": product, "segment": segment, "grade": grade,
                "max_amount": float(max_amount), "max_term": max_term, "max_ratio": round(max_ratio, 4),
                "min_price": round(min_price, 4),
                "cell_id": _cell_id("appetite", APPETITE_VERSION, product, segment, grade),
            })
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "appetite_grid",
        "columns": {"product": "Int64", "segment": "Int64", "grade": "Int64", "max_amount": "Float64",
                    "max_term": "Int64", "max_ratio": "Float64", "min_price": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "product_code", "value_column": "product"},
            {"type": "eq", "variable": "segment_code", "value_column": "segment"},
            {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
        ]},
        "outputs": ["max_amount", "max_term", "max_ratio", "min_price", "cell_id"],
        "default": [0.0, 0, 0.0, 99.0, None],
    })


def binding_appetite_limit(
    requested_amount: float, max_amount: float, term_months: float, max_term: float, max_ratio: float,
    instalment_to_income_ratio: float = missing_as(0.0),
) -> str:
    if requested_amount > max_amount:
        return MAX_AMOUNT
    if term_months > max_term:
        return MAX_TERM
    if instalment_to_income_ratio > max_ratio:
        return MAX_INSTALMENT_RATIO
    return ""


binding_appetite_limit_step = step(binding_appetite_limit)
