"""P12 -- Pricing (spec 10 §5.13): this project's own Flex Loan rate card.

Product 10's card at its **declared non-uniform band shape** (10 §5.13):
amount bands R2 000-R9 999 in R500 steps (16), R10 000-R59 999 in R2 500
(20), R60 000-R199 999 in R10 000 (14), R200 000-R379 999 in R15 000 (12),
R380 000-R500 000 in R12 000 (10) -- 72 bands, exactly as the spec states,
because pricing granularity is deliberately finer at small amounts, and a
uniform-band card would erase the exact non-monotone band-edge behaviour
P13's solve (§5.14) exists to handle. Term columns run 6 to 45 months
(40 discrete columns, one per month, 10 §5.13); terms 46-84 price from the
45-month cell plus a long-term loading table (this project's own, small).
72 x 40 x 12 = 34 560 cells, generated, matching the spec's own declared
count for this card (10 §5.13's table) -- this project's dominant-scale
table, per SCOPE.md's "own tables, own band edges".

Reused from 00 unchanged: `core.fees` (initiation fee, capped, piecewise),
`core.credit_life` (age x term x cover-type premium table -- this
project's own instance is smaller, working depth, same mechanism),
`core.instalment` (amortisation and the fee/premium/instalment ordering,
O-14), `core.rounding`. The rate overlay (10 §5.13(e)) is this project's
own `OVERLAY_REGISTER` entry, applied *after* the cell lookup and *before*
the annuity (O-13: the statutory ceiling is re-checked after the add-on,
not only at card validation).
"""
from __future__ import annotations

from decider import param, step
from decider.steps.tables import DecisionTableConfig

from credit_core.evidence import cell_id as _cell_id
from retail_credit.overlays import ADJUSTMENT_SET_ID, OVERLAY_REGISTER

RATE_CARD_VERSION = "p10-rc-2026.09"
STATUTORY_RATE_CEILING = 0.2825  # 28.25% at a 7.25% repo rate (10 §5.13, illustrative)

_AMOUNT_BAND_STEPS = [
    (2_000.0, 10_000.0, 500.0),      # 16 bands
    (10_000.0, 60_000.0, 2_500.0),   # 20 bands
    (60_000.0, 200_000.0, 10_000.0),  # 14 bands
    (200_000.0, 380_000.0, 15_000.0),  # 12 bands
    (380_000.0, 500_000.0, 12_000.0),  # 10 bands
]


def _amount_bands() -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    for lo, hi, step_size in _AMOUNT_BAND_STEPS:
        edge = lo
        while edge < hi:
            bands.append((edge, min(edge + step_size, hi)))
            edge += step_size
    return bands


def _open_edged(bands: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """+-inf outer edges (not `None`): this table repeats its band ladder once per grade,
    and `DecisionTableConfig` permits exactly one None-lower and one None-upper row in the
    *whole* table (project 00 NOTES.md "Framework friction" §4.5) -- confirmed again here.
    """
    out = list(bands)
    out[0] = (float("-inf"), out[0][1])
    out[-1] = (out[-1][0], float("inf"))
    return out


def _rate_for_cell(amount_index: int, term: int, grade: int) -> float:
    """A synthetic, monotone-in-risk surface with a deliberate non-uniformity at the
    R60 000 band edge (10 §5.14's worked failure: crossing R60 000 saves 155 bp), so this
    project's own card reproduces the exact band-edge inversion P13's solve must handle --
    not merely a monotone toy surface that would let a naive halving search look correct.
    """
    base = 12.45 + (grade - 1) * 1.30
    amount_component = -0.02 * amount_index
    term_component = 0.015 * (term - 6)
    edge_bonus = -1.55 if amount_index >= 36 else 0.0  # band index 36 == the R60 000 band (10 §5.14's cliff)
    rate = base + amount_component + term_component + edge_bonus
    return round(min(max(rate, 12.45), 26.90), 4)


def generate_rate_card(version: str = RATE_CARD_VERSION) -> dict:
    amount_bands = _open_edged(_amount_bands())
    rows = []
    for ai, (amt_lo, amt_hi) in enumerate(amount_bands):
        for term in range(6, 46):
            for grade in range(1, 13):
                rate = _rate_for_cell(ai, term, grade)
                rows.append({
                    "amt_lo": amt_lo, "amt_hi": amt_hi, "term": term, "grade": grade,
                    "rate": rate, "cell_id": _cell_id("p10.rate_card.flex_loan", version, ai, term, grade),
                    "rate_card_version": version,
                })
    return {
        "type": "decision_table", "name": "p10_rate_card_flex_loan",
        "columns": {"amt_lo": "Float64", "amt_hi": "Float64", "term": "Int64", "grade": "Int64",
                    "rate": "Float64", "cell_id": "String", "rate_card_version": "String"},
        "rows": rows,
        "expression": {
            "type": "and",
            "expressions": [
                {"type": "between", "variable": "offered_amount", "lower_bound_column": "amt_lo",
                 "upper_bound_column": "amt_hi", "allow_gaps": True},
                {"type": "eq", "variable": "priced_term_months", "value_column": "term"},
                {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
            ],
        },
        "outputs": ["rate", "cell_id", "rate_card_version"],
        "default": [None, None, None],
    }


CREDIT_LIFE_VERSION = "p10-cl-2026.09"
_INF = float("inf")
# (age_lo, age_hi, term_lo, term_hi, rate per R1000 of cover) -- SINGLE cover only, this
# project's own small table (10 §5.13(c): 16 age bands x 6 term bands x 5 employment types
# = 480 cells in full; working depth here, same mechanism -- three age bands x three term
# bands). Values match the worked example in 10 §5.13 (41-year-old, permanent, 49-60
# months -> R2.95) up to this table's own band edges.
_CREDIT_LIFE_RATES = [
    (-_INF, 30.0, -_INF, 24.0, 3.20), (-_INF, 30.0, 24.0, 60.0, 3.80), (-_INF, 30.0, 60.0, _INF, 4.50),
    (30.0, 55.0, -_INF, 24.0, 4.00), (30.0, 55.0, 24.0, 49.0, 4.70), (30.0, 55.0, 49.0, 60.0, 2.95),
    (30.0, 55.0, 60.0, _INF, 5.60),
    (55.0, _INF, -_INF, 24.0, 6.50), (55.0, _INF, 24.0, 60.0, 7.80), (55.0, _INF, 60.0, _INF, 9.20),
]


def generate_credit_life_table(version: str = CREDIT_LIFE_VERSION) -> dict:
    rows = [
        {"age_lo": age_lo, "age_hi": age_hi, "term_lo": term_lo, "term_hi": term_hi, "rate": rate,
         "cell_id": _cell_id("p10.credit_life", version, i)}
        for i, (age_lo, age_hi, term_lo, term_hi, rate) in enumerate(_CREDIT_LIFE_RATES)
    ]
    return {
        "type": "decision_table", "name": "p10_credit_life_rates",
        "columns": {"age_lo": "Float64", "age_hi": "Float64", "term_lo": "Float64", "term_hi": "Float64",
                    "rate": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "between", "variable": "applicant_age_years", "lower_bound_column": "age_lo",
             "upper_bound_column": "age_hi", "allow_gaps": True},
            {"type": "between", "variable": "offer_term_months", "lower_bound_column": "term_lo",
             "upper_bound_column": "term_hi", "allow_gaps": True},
        ]},
        "outputs": ["rate", "cell_id"],
        "default": [0.0, None],
    }


def load_credit_life_table() -> DecisionTableConfig:
    return DecisionTableConfig.load(generate_credit_life_table())


_CREDIT_LIFE_ROWS = generate_credit_life_table()["rows"]  # single source of truth, shared with the search


def lookup_credit_life_rate(applicant_age_years: float, term_months: int) -> float:
    """The same credit-life table, in pure Python -- **from the identical generated rows**
    `load_credit_life_table()` builds its `DecisionTableConfig` from (the same discipline as
    `lookup_rate_cell`; see its docstring). This is what makes the search's estimate of the
    premium agree with P12's real, table-driven pricing of the chosen offer -- an earlier
    version of this project hardcoded one age band's rate here, which happened to match this
    project's own worked example and silently disagreed with `credit_core.credit_life`'s
    real age/term lookup for every other applicant (`tests/test_solve.py::
    test_price_one_matches_the_wired_credit_life_table` is the regression test for that).
    """
    for row in _CREDIT_LIFE_ROWS:
        if row["age_lo"] <= applicant_age_years < row["age_hi"] and row["term_lo"] <= term_months < row["term_hi"]:
            return row["rate"]
    return 0.0


def load_rate_card() -> DecisionTableConfig:
    return DecisionTableConfig.load(generate_rate_card())


_RATE_ROWS = generate_rate_card()["rows"]  # generated once, shared by the served table and the plain-Python search

# Indexed by (term, grade) -> rows sorted by amt_lo, for the search's bisect lookup below.
_RATE_INDEX: dict[tuple[int, int], list[dict]] = {}
for _row in _RATE_ROWS:
    _RATE_INDEX.setdefault((_row["term"], _row["grade"]), []).append(_row)
for _rows in _RATE_INDEX.values():
    _rows.sort(key=lambda r: r["amt_lo"])


def lookup_rate_cell(offered_amount: float, priced_term: int, grade: int) -> tuple[float, str] | tuple[None, None]:
    """The same card, looked up in pure Python -- **from the identical generated rows**
    `load_rate_card()` builds its `DecisionTableConfig` from, not a re-derivation. P13's
    bounded search (`retail_credit.solve`) calls this directly, up to 19 times per term,
    because invoking a full decider `Engine.score()` per candidate would pay pipeline
    dispatch overhead the search's own 38 ms budget (10 §5.14) cannot afford -- the search
    needs the pricing *function*, not the pricing *step*. Returns `(None, None)` out of
    range, matching the table's own `default`.
    """
    import bisect

    rows = _RATE_INDEX.get((priced_term, grade))
    if not rows:
        return None, None
    lo_edges = [r["amt_lo"] for r in rows]
    i = bisect.bisect_right(lo_edges, offered_amount) - 1
    if i < 0:
        return None, None
    row = rows[i]
    if not (row["amt_lo"] <= offered_amount < row["amt_hi"]):
        return None, None
    return row["rate"] / 100.0, row["cell_id"]


_LONG_TERM_LOADING = {(46, 60): 0.006, (61, 72): 0.011, (73, 84): 0.017}


def priced_term_months(term_months: int) -> int:
    """The card is keyed 6..45; terms 46-84 price from the 45-month column (10 §5.13:
    "terms 46 to 84 are priced from the 45-month column plus a long-term loading")."""
    return min(term_months, 45)


priced_term_months_step = step(priced_term_months)


def long_term_loading(term_months: int) -> float:
    if term_months <= 45:
        return 0.0
    for (lo, hi), loading in _LONG_TERM_LOADING.items():
        if lo <= term_months <= hi:
            return loading
    return _LONG_TERM_LOADING[(73, 84)]


long_term_loading_step = step(long_term_loading)


def rate_card_rate(rate: float | None) -> float | None:
    """Cells are a human-readable percentage (12.45..26.90); the vocabulary field
    `nominal_annual_rate` is a decimal fraction (matches `core.rate_card`'s own convention)."""
    return None if rate is None else rate / 100.0


rate_card_rate_step = step(rate_card_rate, output="nominal_annual_rate_pre_loading")


def add_long_term_loading(nominal_annual_rate_pre_loading: float, long_term_loading: float) -> float:
    return round(nominal_annual_rate_pre_loading + long_term_loading, 6)


add_long_term_loading_step = step(add_long_term_loading, output="nominal_annual_rate_pre_overlay")


def rate_overlay_step():
    """The rate add-on (10 §5.13(e), O-13): applied after the cell lookup (and the
    long-term loading), before the annuity. Same register, resolved once (O-05/O-21).
    """
    return OVERLAY_REGISTER.apply_stack_step(
        "nominal_annual_rate", ADJUSTMENT_SET_ID, base_field="nominal_annual_rate_pre_overlay",
        adjusted_output="nominal_annual_rate", unadjusted_output="nominal_annual_rate_unadjusted",
    )


def _zero() -> float:
    return 0.0


zero_base_step = step(_zero, output="rate_addon_base")


def resolve_rate_addon_bps_step():
    """The rate add-on's basis-point value alone, resolved **once**, before the solve runs
    (O-05/O-21: the same register `rate_overlay_step` reads later for the shipped offer's
    real pricing) -- applying the stack to a base of `0.0` returns exactly the addon amount
    for an `"add"`-kind overlay, which is all `core.adjustments.AdjustmentRegister` needs to
    answer "how much does the solve's own candidate ranking need to move" without
    re-implementing overlay resolution. P13's search (`retail_credit.solve.solve_step`)
    reads this as `overlay_bps`, so a candidate that is only affordable *without* the
    overlay is never returned as the answer (10 §5.13(e): the add-on "participates in the
    solve... including its effect on band-edge behaviour") -- see NOTES.md "Framework
    friction" for the earlier version of this project that skipped this and shipped an
    offer whose real, overlaid instalment exceeded `max_affordable_instalment`.
    """
    return OVERLAY_REGISTER.apply_stack_step(
        "nominal_annual_rate", ADJUSTMENT_SET_ID, base_field="rate_addon_base",
        adjusted_output="rate_addon_bps", unadjusted_output="rate_addon_bps_unadjusted",
    )


def initiation_fee_incl_tax(initiation_fee: float, tax_rate: float = param(0.15, ge=0.0, le=0.25)) -> float:
    """`core.fees.initiation_fee` returns the fee excluding tax (10 §5.13(b): "both
    excluding indirect tax at 15%"); this project's own step for the gross-up, since 00's
    demo pipeline never needed one (it does not capitalise the fee into the financed
    principal -- see `amount_financed` below)."""
    return round(initiation_fee * (1.0 + tax_rate), 2)


initiation_fee_incl_tax_step = step(initiation_fee_incl_tax)


def amount_financed(offered_amount: float, initiation_fee_incl_tax: float) -> float:
    """O-14: the fee is capitalised -- the client receives `offered_amount` and finances
    `offered_amount + fee including tax` (10 §5.13(b)). The annuity, and the credit-life
    premium (which is charged on the amount financed, 10 §5.13(c)), are both computed on
    this, not on `offered_amount` -- the exact distinction `core.instalment`'s own demo
    usage in project 00 does not need to make (project 00's pipeline prices a flat,
    non-capitalised fee; this project's own calibration follows 10 §5.13(b)/(c) literally).
    """
    return round(offered_amount + initiation_fee_incl_tax, 2)


amount_financed_step = step(amount_financed)


def statutory_ceiling_ok(nominal_annual_rate: float, ceiling: float = param(STATUTORY_RATE_CEILING)) -> bool:
    """O-13: re-checked **after** the add-on, not only at card validation -- a card
    validated compliant plus an overlay can still breach the ceiling (10 §5.13)."""
    return nominal_annual_rate <= ceiling


statutory_ceiling_ok_step = step(statutory_ceiling_ok, output="statutory_ceiling_ok")
