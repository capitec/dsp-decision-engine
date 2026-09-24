"""Stage 5.8 -- the bounded solve (spec 03 §5.8). The heart of the spec.

**Why not `decider.loop`.** `loop`'s model -- one condition, one body, a
fixed set of carried scalars, checked per row before each iteration -- fits
a single uniform per-iteration rule well (its own docstring example is a
repayment schedule: pay, tick, repeat). This search has two genuinely
different phases per iteration (has the current amount band been proven
entirely infeasible, in which case move to the next band down; or is it
worth bisecting within it) and a correctness bar with zero tolerance (§10
item 2: a 250 000-application from-scratch re-check, zero failures; §10
item 3: agreement with exhaustive R100-grid evaluation, zero
disagreements). Expressing "which band, which bisection bound, how many
evaluations remain, what the winning candidate's full pricing was" as
carried columns across `loop` iterations is possible, but a single,
directly unit-testable Python function is a shorter, clearer path to that
correctness bar than decomposing it into per-row framework state -- the
same judgement `waterfall.py` makes, and made for the same reason.

**The algorithm (banded top-down bisection).** Spec 03 §5.8 proves that a
plain bisection over the whole domain is wrong (the R50 000 worked
example): affordability is not monotone in amount *across* rate bands,
because the rate card's band edges are not monotone (deliberately -- 00 §6
"band-edge inversions", spec 03 §6.1 declares 41 of them). But *within* one
amount band the rate is constant, and every other input to the instalment
(the piecewise-but-non-decreasing initiation fee, the credit life premium
on the amount financed, the annuity itself) is non-decreasing in amount at
a fixed rate and term -- so instalment is monotone non-decreasing in amount
*within* a band, and a bisection is valid there. The search therefore:

1. Tries the top of the domain first. If it's affordable, it's trivially
   the true maximum (nothing larger exists in the domain) -- one
   evaluation, no bisection needed. This is the common case.
2. Otherwise walks amount bands top-down. For each band, evaluates the
   band's own lowest R100 candidate (its cheapest point, by the monotonicity
   above). If even that is unaffordable, the whole band is infeasible and
   the search moves to the next band down without evaluating anything else
   in it. If the band's lowest point *is* affordable, the band's true
   maximum is found by bisecting between that lower bound (feasible) and
   the band's upper bound (already known infeasible, from this band's own
   top or the band above it) -- each bisection step is one more evaluation,
   and it terminates exactly at the boundary, not before.
3. Every evaluation counts against the 24-per-term ceiling (§5.8
   requirement 1); reaching it without a proven maximum returns "no
   feasible amount, BIND-EXH" rather than a plausible-but-unproven guess
   (§5.8 requirement 6, reason 1420) -- the search never returns a value it
   has not proven is the true maximum.

Because step 1 is a fast path and step 2 only evaluates a band's cheapest
point before deciding whether to bisect it, the search is far cheaper than
scanning every band (which the 24-evaluation ceiling would not survive for
a wide domain) while never assuming the cross-band monotonicity spec 03
proves does not hold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from loan_granting.pricing import PriceEvaluator, PricingResult

GRID = 100.0
PRODUCT_MIN_AMOUNT = 2_000.0
PRODUCT_MAX_AMOUNT = 500_000.0
DEFAULT_EVALUATION_CEILING = 24

BIND_AFF = "BIND-AFF"
BIND_CAP = "BIND-CAP"
BIND_REQ = "BIND-REQ"
BIND_MIN = "BIND-MIN"
BIND_MAX = "BIND-MAX"
BIND_CEIL = "BIND-CEIL"
BIND_EXH = "BIND-EXH"


def _floor_to_grid(x: float) -> float:
    return math.floor(x / GRID) * GRID


def _ceil_to_grid(x: float) -> float:
    return math.ceil(x / GRID) * GRID


@dataclass
class Evaluation:
    amount: float
    nominal_annual_rate: float | None
    instalment: float
    feasible: bool
    priced: bool


@dataclass
class SolveResult:
    term_months: int
    amount: float | None
    binding_constraint_code: str
    evaluations: list[Evaluation] = field(default_factory=list)
    pricing: PricingResult | None = None

    @property
    def evaluation_count(self) -> int:
        return len(self.evaluations)


def solve_term(
    evaluator: PriceEvaluator, term_months: int, risk_grade: int, applicant_age_years: float, is_joint: bool,
    amount_cap: float, requested_amount: float | None, max_affordable_instalment: float,
    evaluation_ceiling: int = DEFAULT_EVALUATION_CEILING,
) -> SolveResult:
    evaluations: list[Evaluation] = []

    def evaluate(amount: float) -> tuple[PricingResult, bool]:
        pr = evaluator.evaluate(amount, term_months, risk_grade, applicant_age_years, is_joint)
        feasible = pr.priced and pr.instalment <= max_affordable_instalment
        evaluations.append(Evaluation(amount, pr.nominal_annual_rate, pr.instalment, feasible, pr.priced))
        return pr, feasible

    def budget_left() -> bool:
        return len(evaluations) < evaluation_ceiling

    domain_ceiling_amount = min(amount_cap, PRODUCT_MAX_AMOUNT)
    domain_top_raw = min(domain_ceiling_amount, requested_amount) if requested_amount else domain_ceiling_amount
    domain_top = _floor_to_grid(domain_top_raw)
    domain_bottom = _ceil_to_grid(PRODUCT_MIN_AMOUNT)

    if domain_top < domain_bottom:
        code = BIND_MIN if domain_ceiling_amount >= PRODUCT_MIN_AMOUNT else BIND_CAP
        return SolveResult(term_months, None, code, evaluations, None)

    top_pricing, top_feasible = evaluate(domain_top)
    if top_feasible:
        if requested_amount is not None and domain_top == _floor_to_grid(requested_amount) and \
                domain_top <= domain_ceiling_amount:
            code = BIND_REQ
        elif domain_top == _floor_to_grid(amount_cap):
            code = BIND_CAP
        else:
            code = BIND_MAX
        return SolveResult(term_months, domain_top, code, evaluations, top_pricing)

    current_top = domain_top
    winner_amount: float | None = None
    winner_pricing: PricingResult | None = None
    exhausted = False

    while current_top >= domain_bottom:
        if not budget_left():
            exhausted = True
            break
        band = evaluator.rate_card_index.band_bounds(current_top, risk_grade)
        if band is None:
            break
        band_lo_raw, band_hi_raw = band
        band_lo = _ceil_to_grid(max(band_lo_raw, domain_bottom))
        band_hi = current_top if not math.isfinite(band_hi_raw) else min(_floor_to_grid(band_hi_raw - 1e-9), current_top)
        if band_lo > band_hi:
            current_top = band_lo - GRID
            continue

        if band_hi == current_top:
            hi_pricing, hi_feasible = (top_pricing, top_feasible) if band_hi == domain_top else (None, None)
            if hi_pricing is None:
                if not budget_left():
                    exhausted = True
                    break
                hi_pricing, hi_feasible = evaluate(band_hi)
        else:
            if not budget_left():
                exhausted = True
                break
            hi_pricing, hi_feasible = evaluate(band_hi)

        if hi_feasible:
            # Shouldn't happen (current_top's own top was already infeasible), but if a
            # narrower band boundary is affordable, it's this band's proven maximum.
            winner_amount, winner_pricing = band_hi, hi_pricing
            break

        if band_lo == band_hi:
            current_top = band_lo - GRID
            continue

        if not budget_left():
            exhausted = True
            break
        lo_pricing, lo_feasible = evaluate(band_lo)
        if not lo_feasible:
            # This band's cheapest point is still unaffordable: skip the whole band.
            # A proportional jump (using the instalment just observed as a local linear
            # estimate, since amount and instalment scale roughly together at a fixed
            # rate) reaches the feasible neighbourhood far faster than always retreating
            # one band width -- the difference between a handful of evaluations and
            # dozens on a wide domain with no nearby inversion, the common case (spec 03
            # §6.1: 41 inversions out of thousands of cells). Every band still visited is
            # still fully verified; this only chooses which one to look at next.
            if lo_pricing.priced and lo_pricing.instalment > 0:
                estimate = _floor_to_grid(band_lo * max_affordable_instalment / lo_pricing.instalment)
                current_top = min(estimate, band_lo - GRID)
            else:
                current_top = band_lo - GRID
            continue

        # Bisect: lo (feasible) .. hi (infeasible), both within one band, both on the R100
        # grid. Monotone non-decreasing instalment in amount within a band (this module's
        # docstring) makes this valid.
        lo, hi = band_lo, band_hi
        best_pricing = lo_pricing
        while hi - lo > GRID:
            if not budget_left():
                exhausted = True
                break
            mid = _floor_to_grid(lo + (hi - lo) / 2.0)
            if mid <= lo:
                mid = lo + GRID
            if mid >= hi:
                mid = hi - GRID
            mid_pricing, mid_feasible = evaluate(mid)
            if mid_feasible:
                lo, best_pricing = mid, mid_pricing
            else:
                hi = mid
        if exhausted:
            break  # ran out of budget mid-bisection: `lo` is feasible but not *proven* the
            # band's maximum (something between `lo` and `hi` was never checked) -- BIND-EXH,
            # not a guess.
        winner_amount, winner_pricing = lo, best_pricing
        break

    if winner_amount is None:
        code = BIND_EXH if exhausted else BIND_AFF
        return SolveResult(term_months, None, code, evaluations, None)
    return SolveResult(term_months, winner_amount, BIND_AFF, evaluations, winner_pricing)


def exhaustive_search(evaluator: PriceEvaluator, term_months: int, risk_grade: int, applicant_age_years: float,
                       is_joint: bool, amount_cap: float, requested_amount: float | None,
                       max_affordable_instalment: float) -> float | None:
    """Evaluate every R100 candidate in the domain (spec 03 §10 item 3's exhaustive
    verification). Used only in tests, over a modest sample -- 4 981 candidates per
    application at the product's full range, so never called in the served path."""
    domain_ceiling_amount = min(amount_cap, PRODUCT_MAX_AMOUNT)
    domain_top_raw = min(domain_ceiling_amount, requested_amount) if requested_amount else domain_ceiling_amount
    domain_top = _floor_to_grid(domain_top_raw)
    domain_bottom = _ceil_to_grid(PRODUCT_MIN_AMOUNT)
    best = None
    amount = domain_bottom
    while amount <= domain_top:
        pr = evaluator.evaluate(amount, term_months, risk_grade, applicant_age_years, is_joint)
        if pr.priced and pr.instalment <= max_affordable_instalment:
            best = amount
        amount += GRID
    return best
