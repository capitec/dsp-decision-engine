"""Product 11 and product 20 rate cards (spec 06 §5.6.4, §5.6.6; §6.1).

Two different *representations* on top of two different key sets, which is
exactly the "rate card cannot be assumed to return a scalar" problem §5.6.6
names:

- **Product 11 (absolute).** grade x amount-band x term-band -> one absolute
  annual rate. Same shape as project 00's Flex Loan card
  (`credit_core.rate_card`), reused directly through 03's `RateCardIndex` /
  `PriceEvaluator` (`consolidation/pricing11.py`) -- a different card, the
  *same* lookup and pricing mechanism, "reused four times over inputs project
  03 never contemplated" (DEPS.md).
- **Product 20 (promotional + reversion).** grade x limit-band -> a
  *promotional* rate for a declared duration, then a *reversion* rate. Two
  small tables, looked up together, returning a structured `(promo_rate,
  promo_duration_months, reversion_rate)` result rather than one scalar.

Both are generated at working depth (12x12x12 / 12x8x5 / 12x20), not the
spec's declared 72x73x12=63 072 / 480 / 240 -- 06's dominant difficulty is
the bounded search (§5.5-5.8), not table size (SCOPE.md rule 1 keeps only
the *one* dominant-difficulty table at full declared size per slice; that
table, for project 03, is the Flex Loan card project 00 already built at
full size). The mechanism -- banded lookup, cell-level attribution, a
reviewable diff -- is identical at any size.
"""
from __future__ import annotations

from credit_core.evidence import cell_id as _cell_id

PRODUCT11_RATE_CARD_VERSION = "rc-p11-2026.09"
PRODUCT20_RATE_CARD_VERSION = "rc-p20-2026.09"

PRODUCT11_MIN_AMOUNT = 10_000.0
PRODUCT11_MAX_AMOUNT = 500_000.0
PRODUCT11_MIN_TERM = 12.0
PRODUCT11_MAX_TERM = 84.0
PRODUCT11_AMOUNT_BANDS = 12
PRODUCT11_TERM_BANDS = 12
PRODUCT11_GRADES = 12

PRODUCT20_MIN_LIMIT = 1_000.0
PRODUCT20_MAX_LIMIT = 300_000.0
PRODUCT20_LIMIT_BANDS = 8
PRODUCT20_GRADES = 12
PRODUCT20_PROMO_DURATIONS = (0, 6, 12, 18, 24)


def _bands(lo: float, hi: float, n: int) -> list[tuple[float, float]]:
    """`n` contiguous bands spanning [lo, hi], +-inf at the outer edges (own copy of
    `credit_core.rate_card`'s private helper -- 00's own is not published in its "What
    I publish" table, so this project does not import it; the shape it produces is
    exactly what `RateCardIndex.band_bounds` needs)."""
    step_size = (hi - lo) / n
    edges = [lo + i * step_size for i in range(n + 1)]
    return [
        (float("-inf") if i == 0 else edges[i], float("inf") if i == n - 1 else edges[i + 1])
        for i in range(n)
    ]


def generate_product11_card(version: str = PRODUCT11_RATE_CARD_VERSION, base_rate: float = 9.0) -> dict:
    """Flex Loan Consolidation (product 11): absolute annual rates, grade dominant, a
    small amount/term drift -- same synthetic shape as `credit_core.rate_card`'s Flex
    Loan card (product 10), because "invented for this repository" applies here too."""
    amount_bands = _bands(PRODUCT11_MIN_AMOUNT, PRODUCT11_MAX_AMOUNT, PRODUCT11_AMOUNT_BANDS)
    term_bands = _bands(PRODUCT11_MIN_TERM, PRODUCT11_MAX_TERM, PRODUCT11_TERM_BANDS)
    rows = []
    for ai, (amt_lo, amt_hi) in enumerate(amount_bands):
        for ti, (term_lo, term_hi) in enumerate(term_bands):
            for grade in range(1, PRODUCT11_GRADES + 1):
                rate = base_rate + (grade - 1) * 1.3 + ai * 0.05 - ti * 0.03
                rate = round(min(max(rate, 8.5), 29.5), 4)
                rows.append({
                    "amt_lo": amt_lo, "amt_hi": amt_hi, "term_lo": term_lo, "term_hi": term_hi,
                    "grade": grade, "rate": rate,
                    "cell_id": _cell_id("rate_card.product11", version, ai, ti, grade),
                    "rate_card_version": version,
                })
    return {
        "type": "decision_table", "name": "rate_card_product11",
        "columns": {"amt_lo": "Float64", "amt_hi": "Float64", "term_lo": "Float64", "term_hi": "Float64",
                    "grade": "Int64", "rate": "Float64", "cell_id": "String", "rate_card_version": "String"},
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


# Margin adjustment by external-creditor proportion (§5.6.4: "4 external-proportion
# bands x 12 grades"), added on top of the base card cell -- basis points, not a
# second absolute rate.
_EXTERNAL_PROPORTION_BANDS = [(0.0, 0.60), (0.60, 0.80), (0.80, 0.95), (0.95, 1.01)]


def margin_adjustment_bps(external_proportion: float, grade: int) -> float:
    for i, (lo, hi) in enumerate(_EXTERNAL_PROPORTION_BANDS):
        if lo <= external_proportion < hi:
            return round((3 - i) * 15 + grade * 2, 1)  # more external -> smaller margin (cheaper refinance)
    return 0.0


def generate_product20_promo_card(version: str = PRODUCT20_RATE_CARD_VERSION, base_rate: float = 6.0) -> dict:
    """Promotional card: grade x limit-band x promotional duration -> promo rate."""
    limit_bands = _bands(PRODUCT20_MIN_LIMIT, PRODUCT20_MAX_LIMIT, PRODUCT20_LIMIT_BANDS)
    rows = []
    for li, (lim_lo, lim_hi) in enumerate(limit_bands):
        for duration in PRODUCT20_PROMO_DURATIONS:
            for grade in range(1, PRODUCT20_GRADES + 1):
                rate = base_rate + (grade - 1) * 0.9 + li * 0.1 - duration * 0.05
                rate = round(min(max(rate, 0.0), 22.0), 4)
                rows.append({
                    "lim_lo": lim_lo, "lim_hi": lim_hi, "duration": duration, "grade": grade,
                    "rate": rate, "cell_id": _cell_id("rate_card.product20.promo", version, li, duration, grade),
                })
    return rows


def generate_product20_reversion_card(version: str = PRODUCT20_RATE_CARD_VERSION, base_rate: float = 21.0) -> dict:
    """Reversion card: grade x limit-band -> reversion rate (the rate after the
    promotional period ends -- the rate the stressed affordability test always uses,
    §5.6.6)."""
    limit_bands = _bands(PRODUCT20_MIN_LIMIT, PRODUCT20_MAX_LIMIT, PRODUCT20_LIMIT_BANDS * 2)
    rows = []
    for li, (lim_lo, lim_hi) in enumerate(limit_bands):
        for grade in range(1, PRODUCT20_GRADES + 1):
            rate = base_rate + (grade - 1) * 1.1 - li * 0.06
            rate = round(min(max(rate, 15.0), 32.0), 4)
            rows.append({
                "lim_lo": lim_lo, "lim_hi": lim_hi, "grade": grade, "rate": rate,
                "cell_id": _cell_id("rate_card.product20.reversion", version, li, grade),
            })
    return rows


class Product20CardIndex:
    """Both product 20 tables, held as one small in-memory index -- the structured
    `(promo_rate, promo_duration_months, reversion_rate)` lookup §5.6.6 requires,
    because "a rate card cannot be assumed to return a scalar" the moment a product
    has two rates and a date."""

    def __init__(self, promo_rows: list[dict], reversion_rows: list[dict], version: str = PRODUCT20_RATE_CARD_VERSION):
        self.promo_rows = promo_rows
        self.reversion_rows = reversion_rows
        self.version = version

    def lookup_promo(self, limit: float, promo_duration_months: int, grade: int) -> tuple[float, str] | None:
        for r in self.promo_rows:
            if r["lim_lo"] <= limit < r["lim_hi"] and r["duration"] == promo_duration_months and r["grade"] == grade:
                return r["rate"] / 100.0, r["cell_id"]
        return None

    def lookup_reversion(self, limit: float, grade: int) -> tuple[float, str] | None:
        for r in self.reversion_rows:
            if r["lim_lo"] <= limit < r["lim_hi"] and r["grade"] == grade:
                return r["rate"] / 100.0, r["cell_id"]
        return None

    @classmethod
    def build(cls) -> "Product20CardIndex":
        return cls(generate_product20_promo_card(), generate_product20_reversion_card())
