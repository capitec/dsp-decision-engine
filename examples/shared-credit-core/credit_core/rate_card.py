"""The Flex Loan rate card: a rate for every amount band, term band and risk grade (96 x 55 x 12 = 63 360 cells).

The card is a data artefact (`configs/<version>/rate_card_flex_loan.json`), so Treasury can replace it
without a code change. `generate_flex_loan_card` builds a synthetic one; `diff_cards` makes a change reviewable.
"""
from __future__ import annotations

from decider import DecisionTableConfig, flow, param, step

from credit_core.evidence import cell_id

FLEX_LOAN_AMOUNT_BANDS = 96
FLEX_LOAN_TERM_BANDS = 55
FLEX_LOAN_GRADES = 12
FLEX_LOAN_MIN_AMOUNT = 2_000.0
FLEX_LOAN_MAX_AMOUNT = 500_000.0
FLEX_LOAN_MIN_TERM = 6.0
FLEX_LOAN_MAX_TERM = 84.0


def _bands(lo: float, hi: float, n: int) -> list[tuple[float, float]]:
    """`n` contiguous bands across `[lo, hi]`. The outer edges are infinite, so a value outside the product's
    range still lands in an edge band (flagged by `out_of_range`) instead of going unpriced.

    The edges are explicit rather than open (`None`) because the ladder repeats for every other band and grade.
    """
    width = (hi - lo) / n
    return [(float("-inf") if i == 0 else lo + i * width, float("inf") if i == n - 1 else lo + (i + 1) * width)
            for i in range(n)]


def _rate(base_rate: float, grade: int, amount_band: int, term_band: int) -> float:
    return round(min(max(base_rate + (grade - 1) * 1.5 + amount_band * 0.01 + term_band * 0.02, 5.0), 32.0), 4)


def generate_flex_loan_card(version: str, base_rate: float = 5.0) -> dict:
    """The card as a `DecisionTableConfig` document with one row per cell.

    The rates are a synthetic surface: they rise with grade, term and amount, between 5% and 32%.
    A row has `rate` (a percentage), `cell_id` and `rate_card_version`. Price a request with `lookup(DecisionTableConfig.load(doc))`.
    """
    amounts = _bands(FLEX_LOAN_MIN_AMOUNT, FLEX_LOAN_MAX_AMOUNT, FLEX_LOAN_AMOUNT_BANDS)
    terms = _bands(FLEX_LOAN_MIN_TERM, FLEX_LOAN_MAX_TERM, FLEX_LOAN_TERM_BANDS)
    rows = [
        {"amt_lo": amt_lo, "amt_hi": amt_hi, "term_lo": term_lo, "term_hi": term_hi, "grade": grade,
         "amt_band": ai, "term_band": ti, "rate": _rate(base_rate, grade, ai, ti),
         "cell_id": cell_id("rate_card.flex_loan", version, ai, ti, grade), "rate_card_version": version}
        for ai, (amt_lo, amt_hi) in enumerate(amounts)
        for ti, (term_lo, term_hi) in enumerate(terms)
        for grade in range(1, FLEX_LOAN_GRADES + 1)
    ]
    return {
        "type": "decision_table", "name": "rate_card_flex_loan",
        "columns": {"amt_lo": "Float64", "amt_hi": "Float64", "term_lo": "Float64", "term_hi": "Float64",
                    "grade": "Int64", "amt_band": "Int64", "term_band": "Int64", "rate": "Float64",
                    "cell_id": "String", "rate_card_version": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "between", "variable": "offered_amount", "lower_bound_column": "amt_lo",
             "upper_bound_column": "amt_hi", "allow_gaps": True},
            {"type": "between", "variable": "term_months", "lower_bound_column": "term_lo",
             "upper_bound_column": "term_hi", "allow_gaps": True},
            {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
        ]},
        "outputs": ["rate", "cell_id", "rate_card_version"],
        "default": [None, None, None],
    }


def rate_card_rate(rate: float) -> float:
    """The card holds percentages (easier for Treasury to author and diff); the vocabulary's `nominal_annual_rate` is a fraction."""
    return rate / 100.0


rate_card_rate_step = step(rate_card_rate, output="nominal_annual_rate")


def lookup(card: DecisionTableConfig):
    """The card, then the steps that write `nominal_annual_rate` (a fraction), `rate_cell_id` and `rate_card_version`.

    Reads `offered_amount`, `term_months` and `risk_grade`.
    """
    return flow(card.relabel(writes={"cell_id": "rate_cell_id"}), rate_card_rate_step, name="rate_card").emit(
        "nominal_annual_rate", "rate_cell_id", "rate_card_version")


def out_of_range(
    offered_amount: float, term_months: float,
    min_amount: float = param(FLEX_LOAN_MIN_AMOUNT), max_amount: float = param(FLEX_LOAN_MAX_AMOUNT),
    min_term: float = param(FLEX_LOAN_MIN_TERM), max_term: float = param(FLEX_LOAN_MAX_TERM),
) -> bool:
    """Whether the request lies outside the product's range (the card still prices it from its edge band)."""
    return not (min_amount <= offered_amount <= max_amount and min_term <= term_months <= max_term)


def diff_cards(old_rows: list[dict], new_rows: list[dict], key_fields=("amt_lo", "term_lo", "grade")) -> list[dict]:
    """Which cells were added, removed or repriced between two cards, in key order.

    Example: `diff_cards(old["rows"], new["rows"])` -> `[{"key": (...), "change": "repriced", "delta": 0.25, ...}]`
    """
    old = {tuple(r[k] for k in key_fields): r for r in old_rows}
    new = {tuple(r[k] for k in key_fields): r for r in new_rows}
    changes = []
    for key in sorted(old.keys() | new.keys()):
        before, after = old.get(key), new.get(key)
        if before is None:
            changes.append({"key": key, "change": "added", "rate": after["rate"]})
        elif after is None:
            changes.append({"key": key, "change": "removed", "rate": before["rate"]})
        elif before["rate"] != after["rate"]:
            changes.append({"key": key, "change": "repriced", "old_rate": before["rate"], "new_rate": after["rate"],
                            "delta": round(after["rate"] - before["rate"], 4)})
    return changes
