"""Risk appetite: the limits for a product, segment and risk grade, and which one a request breaks first."""
from __future__ import annotations

from decider import DecisionTableConfig, missing_as, step

from credit_core.evidence import cell_id

APPETITE_VERSION = "appetite-2026.09"

MAX_AMOUNT = "max_amount"
MAX_TERM = "max_term"
MAX_INSTALMENT_RATIO = "max_instalment_to_income_ratio"

# (product, segment) -> (top amount, amount lost per grade, max term, top ratio, ratio lost per grade, top price)
_GRID = {(10, 1): (500_000, 35_000, 84, 0.45, 0.03, 5.0), (10, 2): (450_000, 32_000, 72, 0.40, 0.03, 5.5)}


def build_appetite_table() -> DecisionTableConfig:
    """Writes `max_amount`, `max_term`, `max_ratio`, `min_price` and `cell_id` for `product_code`, `segment_code` and `risk_grade`.

    A combination with no row gets zero limits, so nothing is within appetite.
    """
    rows = [{"product": product, "segment": segment, "grade": grade,
             "max_amount": float(amount - grade * amount_step), "max_term": term,
             "max_ratio": round(max(0.05, ratio - grade * ratio_step), 4), "min_price": round(price + grade * 1.5, 4),
             "cell_id": cell_id("appetite", APPETITE_VERSION, product, segment, grade)}
            for (product, segment), (amount, amount_step, term, ratio, ratio_step, price) in _GRID.items()
            for grade in range(1, 13)]
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
    """The first limit the request breaks (amount, then term, then ratio); empty when none."""
    if requested_amount > max_amount:
        return MAX_AMOUNT
    if term_months > max_term:
        return MAX_TERM
    if instalment_to_income_ratio > max_ratio:
        return MAX_INSTALMENT_RATIO
    return ""


binding_appetite_limit_step = step(binding_appetite_limit)
