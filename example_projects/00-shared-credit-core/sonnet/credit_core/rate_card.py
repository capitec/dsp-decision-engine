"""`core.rate_card` -- priced rate lookup (spec 00 §6.9, §8).

A generic keyed lookup over a priced grid: `product_code`, `offered_amount`,
`term_months`, `risk_grade`, plus product-specific extra keys. The Flex
Loan card is generated at its full declared size, 96 amount bands x 55
term bands x 12 grades = 63 360 cells (00 §8), because shrinking the
dominant-difficulty table defeats the point of this slice (SCOPE.md rule 1).

Each row carries a content-derived `cell_id` and the card's `version`, so
"which cell of which version" (09 §5.15 item 5) is a lookup output, not a
side channel. `diff_cards` is the reviewable-diff artefact §8 requires for
every table change ("Treasury sent a new spreadsheet" is not reviewable).
"""
from __future__ import annotations

from decider import param, step
from decider.steps.tables import DecisionTableConfig

from credit_core.evidence import cell_id as _cell_id

FLEX_LOAN_AMOUNT_BANDS = 96
FLEX_LOAN_TERM_BANDS = 55
FLEX_LOAN_GRADES = 12
FLEX_LOAN_MIN_AMOUNT = 2_000.0
FLEX_LOAN_MAX_AMOUNT = 500_000.0
FLEX_LOAN_MIN_TERM = 6.0
FLEX_LOAN_MAX_TERM = 84.0


def _bands(lo: float, hi: float, n: int) -> list[tuple[float, float]]:
    """`n` contiguous bands spanning [lo, hi], open-ended at both outer edges via
    +-inf sentinels (so a value outside the product's nominal range still lands
    in an edge cell, flagged out-of-range by the caller rather than left unpriced).

    Plain `None` open edges only work for a single unbroken ladder of rows
    (`DecisionTableConfig` requires exactly one None-lower row and one
    None-upper row in the *whole* table); this table repeats the same band
    ladder once per grade, so every group needs its own finite-looking edge.
    """
    step_size = (hi - lo) / n
    edges = [lo + i * step_size for i in range(n + 1)]
    bands = []
    for i in range(n):
        band_lo = float("-inf") if i == 0 else edges[i]
        band_hi = float("inf") if i == n - 1 else edges[i + 1]
        bands.append((band_lo, band_hi))
    return bands


def generate_flex_loan_card(version: str, base_rate: float = 5.0) -> dict:
    """The Flex Loan rate card as a `DecisionTableConfig` document, at full declared size.

    Rates are a synthetic, monotone-in-risk illustrative surface (front
    matter: every number in this repo is invented) -- grade dominates,
    with small amount- and term-band adjustments, clipped to a plausible
    unsecured-lending range.
    """
    amount_bands = _bands(FLEX_LOAN_MIN_AMOUNT, FLEX_LOAN_MAX_AMOUNT, FLEX_LOAN_AMOUNT_BANDS)
    term_bands = _bands(FLEX_LOAN_MIN_TERM, FLEX_LOAN_MAX_TERM, FLEX_LOAN_TERM_BANDS)
    rows = []
    for ai, (amt_lo, amt_hi) in enumerate(amount_bands):
        for ti, (term_lo, term_hi) in enumerate(term_bands):
            for grade in range(1, FLEX_LOAN_GRADES + 1):
                rate = base_rate + (grade - 1) * 1.5 + ai * 0.01 + ti * 0.02
                rate = round(min(max(rate, 5.0), 32.0), 4)
                rows.append({
                    "amt_lo": amt_lo, "amt_hi": amt_hi,
                    "term_lo": term_lo, "term_hi": term_hi,
                    "grade": grade,
                    "rate": rate,
                    "cell_id": _cell_id("rate_card.flex_loan", version, ai, ti, grade),
                    "rate_card_version": version,
                })
    return {
        "type": "decision_table",
        "name": "rate_card_flex_loan",
        "columns": {
            "amt_lo": "Float64", "amt_hi": "Float64", "term_lo": "Float64", "term_hi": "Float64",
            "grade": "Int64", "rate": "Float64", "cell_id": "String", "rate_card_version": "String",
        },
        "rows": rows,
        "expression": {
            "type": "and",
            "expressions": [
                {"type": "between", "variable": "offered_amount", "lower_bound_column": "amt_lo",
                 "upper_bound_column": "amt_hi", "allow_gaps": True},
                {"type": "between", "variable": "term_months", "lower_bound_column": "term_lo",
                 "upper_bound_column": "term_hi", "allow_gaps": True},
                {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
            ],
        },
        "outputs": ["rate", "cell_id", "rate_card_version"],
        "default": [None, None, None],
    }


def load_flex_loan_card(path: str) -> DecisionTableConfig:
    return DecisionTableConfig.load(path)


def rate_card_rate(rate: float | None) -> float | None:
    """The table's `rate` cells are a human-readable percentage (5.0..32.0, easier for
    Treasury to author and diff, 00 §8); the published vocabulary field `nominal_annual_rate`
    (00 §4) is a decimal fraction, the convention `core.instalment` and every rate-consuming
    capability in this library expects. This is the one place that conversion happens.
    """
    return None if rate is None else rate / 100.0


rate_card_rate_step = step(rate_card_rate, output="nominal_annual_rate")


def out_of_range(offered_amount: float, term_months: float,
                  min_amount: float = param(FLEX_LOAN_MIN_AMOUNT), max_amount: float = param(FLEX_LOAN_MAX_AMOUNT),
                  min_term: float = param(FLEX_LOAN_MIN_TERM), max_term: float = param(FLEX_LOAN_MAX_TERM)) -> bool:
    """True when the record falls outside the card's nominal product range (00 §6.9 "Hard part")."""
    return not (min_amount <= offered_amount <= max_amount and min_term <= term_months <= max_term)


def diff_cards(old_rows: list[dict], new_rows: list[dict], key_fields=("amt_lo", "term_lo", "grade")) -> list[dict]:
    """The reviewable diff between two card generations: which cells changed, by how much (00 §8).

    A cell present in one side only is a structural change (band count
    changed); a cell present in both with a different `rate` is a repricing.
    """

    def key(row: dict) -> tuple:
        return tuple(row[k] for k in key_fields)

    old_by_key = {key(r): r for r in old_rows}
    new_by_key = {key(r): r for r in new_rows}
    changes = []
    for k in old_by_key.keys() | new_by_key.keys():
        before = old_by_key.get(k)
        after = new_by_key.get(k)
        if before is None:
            changes.append({"key": k, "change": "added", "rate": after["rate"]})
        elif after is None:
            changes.append({"key": k, "change": "removed", "rate": before["rate"]})
        elif before["rate"] != after["rate"]:
            changes.append({"key": k, "change": "repriced", "old_rate": before["rate"], "new_rate": after["rate"],
                             "delta": round(after["rate"] - before["rate"], 4)})
    return changes
