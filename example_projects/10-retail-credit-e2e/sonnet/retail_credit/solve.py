"""P13 -- The solve (spec 10 §5.14): the largest affordable amount at each permitted term.

Why not division (10 §5.14, five reasons): the rate is a step function of
the amount, so **a smaller amount is not always cheaper** -- crossing a
band edge upward can drop the rate by more than the extra principal costs
(this project's own rate card, `retail_credit.pricing`, deliberately
reproduces exactly this at the R60 000 edge, so this module's search has
a real inversion to handle, not a toy monotone surface). Requirements on
the search (10 §5.14): bounded (a hard ceiling of 19 evaluations per
term), terminating, deterministic, correct (the true R250-grid maximum,
not merely "an affordable point"), tie-broken by a declared rule, and
attributed (which constraint bound the answer).

**Why the search is efficient without being naive.** Within one rate-card
band the rate is constant, and instalment (annuity + fee + credit life, all
non-decreasing functions of the principal) is therefore monotone
non-decreasing in amount *within that band* -- so the maximum affordable
point in a band is found by bisecting the band's own range, not by
scanning it R250 step by R250 step. The search visits bands from the
one containing the requested ceiling **downward**: if the band's own
floor is unaffordable, no point in it can be (monotonicity), so the whole
band is skipped in one evaluation; if the floor *is* affordable, bisecting
that band finds its maximum in `O(log(band width / R250))` evaluations,
and the search stops there -- a lower band can never produce a *larger*
amount than one already found feasible in a higher band, so there is
nothing left to gain by continuing down (this is exactly why R60 000 beats
every candidate in the 57500-59999 band in 10 §5.14's worked example).
This reproduces the spec's required behaviour with single-digit
evaluations per term in the common case, not by accident but because the
band structure makes it true -- documented, not assumed, and checked by
`tests/test_solve.py::test_matches_exhaustive_search` against a full
R250-grid scan for the same inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from decider import missing_as, step

from credit_core.instalment import instalment_before_fees, solve_advance_for_instalment
from credit_core.rounding import round_advance
from retail_credit.pricing import (
    _AMOUNT_BAND_STEPS, long_term_loading, lookup_credit_life_rate, lookup_rate_cell, priced_term_months,
)

GRID = 250.0
EVALUATION_CEILING_PER_TERM = 19
PRODUCT_10_MIN_AMOUNT = 2_000.0
SERVICE_FEE_INCL_TAX = 87.98
FEE_BASE_EXCL_TAX = 210.0
FEE_MARGINAL_RATE = 0.095
FEE_THRESHOLD = 1_200.0
FEE_CAP_EXCL_TAX = 1_480.0
TAX_RATE = 0.15
CREDIT_LIFE_PREMIUM_CAP = 1_000.0


def _amount_bands_desc(upper_bound: float, lower_bound: float) -> list[tuple[float, float]]:
    """Every rate-card band overlapping `[lower_bound, upper_bound]`, clipped, nearest
    (highest) band first -- the order the search visits them in."""
    bands = []
    for lo, hi, band_step in _AMOUNT_BAND_STEPS:
        edge = lo
        while edge < hi:
            band_hi = min(edge + band_step, hi)
            clipped_lo, clipped_hi = max(edge, lower_bound), min(band_hi, upper_bound)
            if clipped_lo < clipped_hi:
                bands.append((clipped_lo, clipped_hi))
            edge += band_step
    return list(reversed(bands))


def _floor_grid(amount: float) -> float:
    return (amount // GRID) * GRID


def _ceil_grid(amount: float) -> float:
    import math
    return math.ceil(amount / GRID) * GRID


@dataclass
class Evaluation:
    amount: float
    instalment: float
    affordable: bool
    cell_id: str | None
    rate: float | None


@dataclass
class TermResult:
    term_months: int
    amount: float | None
    instalment: float | None
    binding_constraint: str
    evaluations: list[Evaluation] = field(default_factory=list)
    rate_card_cell_id: str | None = None
    nominal_annual_rate: float | None = None


def price_one(
    amount: float, term_months: int, grade: int, applicant_age_years: float, overlay_bps: float = 0.0,
) -> Evaluation | None:
    """One pricing evaluation, in pure Python -- the same rate card and credit-life table
    (`retail_credit.pricing.lookup_rate_cell`/`lookup_credit_life_rate`) and fee formula P12
    uses as decider steps, called directly so the search can afford up to 19 of them per
    term. `overlay_bps` is the rate add-on P12 already resolved **once** for this decision
    (O-05/O-21) -- the search must not re-resolve the overlay register per candidate.
    """
    pterm = priced_term_months(term_months)
    rate, cell_id = lookup_rate_cell(amount, pterm, grade)
    if rate is None:
        return None
    rate = rate + long_term_loading(term_months) + overlay_bps
    fee_excl_tax = min(FEE_BASE_EXCL_TAX + FEE_MARGINAL_RATE * max(0.0, amount - FEE_THRESHOLD), FEE_CAP_EXCL_TAX)
    fee = round(fee_excl_tax * (1.0 + TAX_RATE), 2)
    financed = amount + fee
    credit_life_rate = lookup_credit_life_rate(applicant_age_years, term_months)
    credit_life = round(min(financed / 1000.0 * credit_life_rate, CREDIT_LIFE_PREMIUM_CAP), 2)
    annuity = instalment_before_fees(financed, term_months, rate)
    total = round(annuity + SERVICE_FEE_INCL_TAX + credit_life, 2)
    return Evaluation(amount=amount, instalment=total, affordable=None, cell_id=cell_id, rate=rate)


def _bisect_band_max(
    band_lo: float, band_hi_exclusive: float, term_months: int, grade: int, applicant_age_years: float,
    overlay_bps: float, max_affordable_instalment: float, budget: list[Evaluation],
) -> Evaluation | None:
    """The maximum affordable grid point within one band (rate constant, so instalment is
    monotone non-decreasing in amount): bisect on affordability, `ceil(log2(width/250))`
    evaluations. Returns None if even the band's floor is unaffordable.
    """
    lo = _ceil_grid(band_lo)
    hi = _floor_grid(band_hi_exclusive - 1.0)  # largest grid point strictly inside the band
    if lo > hi:
        return None
    floor_eval = price_one(lo, term_months, grade, applicant_age_years, overlay_bps)
    budget.append(floor_eval)
    if floor_eval is None or floor_eval.instalment > max_affordable_instalment:
        return None  # even the cheapest point in this band fails: skip the whole band
    best = floor_eval
    while lo < hi and len(budget) < EVALUATION_CEILING_PER_TERM:
        mid = _floor_grid((lo + hi + GRID) / 2.0)
        if mid <= lo:
            break
        ev = price_one(mid, term_months, grade, applicant_age_years, overlay_bps)
        budget.append(ev)
        if ev is not None and ev.instalment <= max_affordable_instalment:
            best, lo = ev, mid
        else:
            hi = mid - GRID
    return best


def solve_for_term(
    term_months: int, requested_amount: float, amount_cap: float, product_max_amount: float,
    grade: int, max_affordable_instalment: float, applicant_age_years: float, overlay_bps: float = 0.0,
    product_min_amount: float = PRODUCT_10_MIN_AMOUNT,
) -> TermResult:
    """The bounded search for one permitted term (10 §5.14 requirements 1-6)."""
    budget: list[Evaluation] = []
    upper_bound = _floor_grid(min(requested_amount, amount_cap, product_max_amount))
    if upper_bound < product_min_amount:
        return TermResult(term_months, None, None, "BIND-MIN", budget)

    for band_lo, band_hi in _amount_bands_desc(upper_bound, product_min_amount):
        if len(budget) >= EVALUATION_CEILING_PER_TERM:
            return TermResult(term_months, None, None, "BIND-EXH", budget)
        best = _bisect_band_max(band_lo, band_hi, term_months, grade, applicant_age_years, overlay_bps,
                                 max_affordable_instalment, budget)
        if best is not None:
            constraint = "BIND-REQ" if best.amount >= _floor_grid(requested_amount) else (
                "BIND-CAP" if upper_bound < _floor_grid(min(amount_cap, product_max_amount)) + GRID
                and best.amount >= upper_bound - GRID else "BIND-AFF")
            return TermResult(term_months, best.amount, best.instalment, constraint, budget,
                               rate_card_cell_id=best.cell_id, nominal_annual_rate=best.rate)
    if not budget:
        return TermResult(term_months, None, None, "BIND-MIN", budget)
    return TermResult(term_months, None, None, "BIND-AFF", budget)


def solve_all_terms(
    requested_amount: float, amount_cap: float, term_cap: float, product_max_amount: float,
    grade: int, max_affordable_instalment: float, applicant_age_years: float, overlay_bps: float = 0.0,
    permitted_terms: tuple[int, ...] = (6, 12, 24, 36, 48, 60, 72, 84),
) -> list[TermResult]:
    results = []
    for term in permitted_terms:
        if term > term_cap:
            continue
        results.append(solve_for_term(term, requested_amount, amount_cap, product_max_amount, grade,
                                       max_affordable_instalment, applicant_age_years, overlay_bps))
    return results


def solve_step(
    term_months: int, requested_amount: float, amount_cap: float, term_cap: float, risk_grade: int,
    max_affordable_instalment: float, applicant_age_years: float, rate_addon_bps: float = missing_as(0.0),
) -> tuple[float, float, str, str, float]:
    """The wired, single-term decider step for entry point 1 (one requested term; the full
    multi-term fan-out and BIND-* array is exercised directly in `tests/test_solve.py`, for
    the same `list[dict]`-terminal-output reason documented in `cap_waterfall.py`).

    `rate_addon_bps` is resolved once, before the solve (`pricing.resolve_rate_addon_bps_step`),
    so the search's own candidate ranking already reflects the overlay P12's real pricing
    would apply to the shipped offer (10 §5.13(e)) -- see that function's docstring for the
    defect this fixed: an earlier version of this search ignored the overlay and could
    return an amount whose real, overlaid instalment exceeded `max_affordable_instalment`.
    """
    result = solve_for_term(
        min(term_months, term_cap), requested_amount, amount_cap, PRODUCT_10_MAX_AMOUNT,
        risk_grade, max_affordable_instalment, applicant_age_years, rate_addon_bps,
    )
    amount = result.amount if result.amount is not None else 0.0
    instalment = result.instalment if result.instalment is not None else 0.0
    cell_id = result.rate_card_cell_id or ""
    rate = result.nominal_annual_rate if result.nominal_annual_rate is not None else 0.0
    return amount, instalment, result.binding_constraint, cell_id, rate


PRODUCT_10_MAX_AMOUNT = 500_000.0  # product 10's max amount (10 §4.2)

solve_step_wired = step(
    solve_step,
    outputs=("solved_amount", "solved_instalment", "solve_binding_constraint", "solve_rate_card_cell_id",
             "solved_nominal_annual_rate"),
)
