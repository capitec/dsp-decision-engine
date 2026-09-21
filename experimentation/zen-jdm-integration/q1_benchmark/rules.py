"""The one rule set, defined once, consumed by both engines.

Q1 asks whether ZEN could be decider2's execution engine, and what it would
cost. To answer that fairly, both engines must run the literal same rules —
not "a" decision table and "a similar" JDM document, but one Python data
structure translated mechanically into each engine's native shape.

The shape: a single-column banded lookup (income -> tier, limit), 12 rows,
contiguous and sorted. This is deliberately the shape both engines handle
natively and well — decider2's `BetweenExpression` (doc 08 §3.4, the "editing
rows is free" table) and ZEN/JDM's `decisionTableNode` with targeted range
unary tests (the same shape as the ZEN repo's own `test-data/table.json` and
`credit-analysis.json`'s "Turnover" node). It is not a contrived worst case
for either side.

`tier` is an int code (0..4), not a string. decider2 hoists string output
literals through a row-index decode step (`TableModule.decode`) that only
applies to the batch/frame path; keeping the comparison on plain int64/
float64 outputs avoids that machinery on the decider2 side without changing
what is being measured (ZEN still emits the string; TIER_NAMES below is the
mapping used to assert the two engines agree).
"""
from __future__ import annotations

TIER_NAMES = ["declined", "low", "standard", "standard", "premium", "premium", "elite", "elite"]

# (lower, upper, tier_code, tier_name, limit) — upper=None means open/last row.
# lower of row 0 is implicitly open (None); every other row's lower is the
# previous row's upper, enforced identically by both engines' validators.
BANDS = [
    (None, 5_000, 0, "declined", 0.0),
    (5_000, 8_000, 1, "low", 5_000.0),
    (8_000, 12_000, 1, "low", 8_000.0),
    (12_000, 17_000, 2, "standard", 15_000.0),
    (17_000, 23_000, 2, "standard", 20_000.0),
    (23_000, 30_000, 2, "standard", 30_000.0),
    (30_000, 40_000, 3, "premium", 50_000.0),
    (40_000, 55_000, 3, "premium", 75_000.0),
    (55_000, 75_000, 3, "premium", 100_000.0),
    (75_000, 100_000, 4, "elite", 150_000.0),
    (100_000, 150_000, 4, "elite", 250_000.0),
    (150_000, None, 4, "elite", 500_000.0),
]


def oracle(income: float) -> tuple[int, float]:
    """The independent reference implementation neither engine is built from.

    Both `build_decider2.py`'s DecisionTable and `build_zen.py`'s JDM
    document are translations of `BANDS`. This is a *third*, independent
    computation of the same function, written directly against the
    half-open [lower, upper) convention both engines were configured to use
    (BoundMode.lower_inclusive on the decider2 side; `[a..b)` cells on the
    ZEN side, confirmed by direct probe against
    `zen.evaluate_unary_expression` before this file was written). Used to
    catch the case where both engines are wrong in the *same* way because
    they were built from the same buggy translation.
    """
    for lower, upper, code, _name, limit in BANDS:
        lo_ok = lower is None or income >= lower
        hi_ok = upper is None or income < upper
        if lo_ok and hi_ok:
            return code, limit
    raise AssertionError(f"income {income} matched no band — BANDS has a gap")
