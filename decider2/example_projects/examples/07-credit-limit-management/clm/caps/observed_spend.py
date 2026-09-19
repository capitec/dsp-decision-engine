"""C5, the observed-spend cap -- the most frequently binding cap in the
programme (384 000 accounts) and the one that stops it doing something
indefensible.

A client with a R6 000 limit, a clean record and 40 months on book sits in a
1.6 cell. The matrix takes them to R9 600. If their highest spending month in
six was R400, R9 600 is exposure the Bank is creating for nobody's benefit.
This cap holds them at their current limit.

Shown in full because it is the shape of a rule that a policy analyst must be
able to read in the generated artefact: two params with bounds, one table
lookup by spend band, one floor, and a docstring that is the reviewable prose.
"""

from decider2 import module, param, table
from decider2.credit import overlay_point

SpendCapParameters = table(
    "clm.spend_cap_parameters",
    keys=("product_code", "spend_band"),
    values={"multiple": float, "floor_c": int},
    domain={"product_code": [20, 21], "spend_band": range(1, 6)},
    unit={"floor_c": "ZAR"}, scale={"floor_c": 100},
    dense=True,
    effective_dated=True,
    owner="Credit Risk Policy",
)


def spend_band(observed_spend_p90_c: int, tables) -> int:
    """Which of five spend bands the trailing 90th-percentile month falls in."""
    return tables.spend_bands.index(observed_spend_p90_c)


def observed_spend_multiple(product_code: int, spend_band: int, tables) -> float:
    """The cap multiple for this product and spend band. 3.5 across the book at
    2026Q3; Credit Risk Policy re-tunes it quarterly by band."""
    return tables.spend_cap_parameters.multiple[
        tables.spend_cap_parameters.cell(product_code, spend_band)]


ApplySpendCapOverlays = overlay_point(
    "spend_cap_overlays",
    register="overlay_set",
    adjusts={"observed_spend_multiple": "multiply"},
    scope_keys=["product_code", "spend_band", "employment_type_code"],
    order="declared",
)


def cap_value_c(
    observed_spend_p90_c: int,
    observed_spend_multiple: float,
    current_limit_c: int,
    product_code: int,
    spend_band: int,
    tables,
) -> int:
    """C5 - the limit may not exceed the greater of the current limit and a
    multiple of the client's trailing 6-month 90th-percentile monthly purchase
    value, floored at R2 000. A client who has never spent more than R400 in a
    month does not receive a R30 000 limit."""
    pass  # max(current_limit_c, round_half_up(p90 * multiple), floor_c)


ObservedSpendCap = module(
    spend_band, observed_spend_multiple, cap_value_c,
    name="observed_spend",
    evidence=["observed_spend_p90_c", "observed_spend_multiple", "spend_band"],
) | ApplySpendCapOverlays
