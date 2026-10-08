"""The bounded solve: the largest amount a client can afford at one term.

There is no formula. The rate is a step function of the amount, so a smaller amount can land in a worse band and
cost a *higher* instalment; the advance rounds to R100; the fee is capitalised. So the instalment is not monotone in
the amount and a plain bisection can under-lend.

What does hold: while the rate never falls as the amount rises, the instalment never falls either (the fee, the
premium and the annuity are all non-decreasing). So the amount range splits at every place the card's rate *drops*
into stretches on which affordability is a prefix, and each stretch is bisected on its own, highest first. The first
stretch with an affordable amount holds the true maximum, which is why the answer always equals `exhaustive_search`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from loan_granting.pricing import INF, PriceEvaluator, PricingResult

GRID = 100.0
PRODUCT_MIN_AMOUNT = 2_000.0
PRODUCT_MAX_AMOUNT = 500_000.0
DEFAULT_EVALUATION_CEILING = 24

BIND_AFF = "BIND-AFF"  # the instalment the client can afford
BIND_CAP = "BIND-CAP"  # the policy amount cap
BIND_REQ = "BIND-REQ"  # the amount asked for
BIND_MIN = "BIND-MIN"  # the product minimum
BIND_MAX = "BIND-MAX"  # the product maximum
BIND_CEIL = "BIND-CEIL"  # no cell at or below the statutory rate ceiling
BIND_EXH = "BIND-EXH"  # the evaluation ceiling was reached before the maximum was proven


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


class _CeilingReached(Exception):
    pass


def _floor_grid(x: float) -> int:
    return math.floor(x / GRID)


def _ceil_grid(x: float) -> int:
    return math.ceil(x / GRID)


def _stretches(evaluator: PriceEvaluator, term_months: int, risk_grade: int, bottom: int, top: int) -> list[tuple[int, int]]:
    """Grid positions (first, last) of each stretch on which the rate never falls, lowest first.

    Stretches that cannot be priced at all are dropped, and a stretch ends at its last priced band.
    """
    bands = evaluator.band_rates(term_months, risk_grade)
    stretches: list[list] = []  # [first, last priced, rate of the latest band]
    for i, (start, rate) in enumerate(bands):
        first = _ceil_grid(max(start, bottom * GRID))
        last = min(_ceil_grid(bands[i + 1][0]) - 1, top) if i + 1 < len(bands) else top
        if first > last:
            continue
        if not stretches or rate < stretches[-1][2]:
            stretches.append([first, None, rate])
        stretches[-1][2] = rate
        if rate < INF:
            stretches[-1][1] = last
    return [(first, last) for first, last, _ in stretches if last is not None]


def solve_term(
    evaluator: PriceEvaluator, term_months: int, risk_grade: int, applicant_age_years: float, is_joint: bool,
    amount_cap: float, requested_amount: float | None, max_affordable_instalment: float,
    evaluation_ceiling: int = DEFAULT_EVALUATION_CEILING,
) -> SolveResult:
    """The largest R100 amount up to the cap and the request whose instalment fits, at `term_months`.

    At most `evaluation_ceiling` candidates are priced. If that is not enough to prove the maximum the result has
    no amount and `BIND-EXH`; it never returns an amount it has not proven.

        result = solve_term(evaluator, 60, 9, 38.0, False, 500_000.0, None, 1_560.0)
        result.amount, result.binding_constraint_code   # (50000.0, 'BIND-AFF')
    """
    evaluations: list[Evaluation] = []

    def probe(position: int) -> tuple[PricingResult, bool]:
        if len(evaluations) >= evaluation_ceiling:
            raise _CeilingReached
        pricing = evaluator.evaluate(position * GRID, term_months, risk_grade, applicant_age_years, is_joint)
        affordable = pricing.priced and pricing.instalment <= max_affordable_instalment
        evaluations.append(Evaluation(position * GRID, pricing.nominal_annual_rate, pricing.instalment, affordable,
                                      pricing.priced))
        return pricing, affordable

    limits = [(BIND_CAP, amount_cap), (BIND_MAX, PRODUCT_MAX_AMOUNT)]
    if requested_amount is not None and requested_amount > 0:
        limits.insert(0, (BIND_REQ, requested_amount))
    top = _floor_grid(min(value for _, value in limits))
    bottom = _ceil_grid(PRODUCT_MIN_AMOUNT)
    if top < bottom:
        return SolveResult(term_months, None, BIND_CAP if amount_cap < PRODUCT_MIN_AMOUNT else BIND_MIN)

    stretches = _stretches(evaluator, term_months, risk_grade, bottom, top)
    try:
        for first, last in reversed(stretches):
            found = _largest_affordable(probe, first, last, last == top)
            if found:
                position, pricing = found
                # When the domain top is affordable, the limit that put it there is what bound the amount.
                code = next(c for c, v in limits if _floor_grid(v) == top) if position == top else BIND_AFF
                return SolveResult(term_months, position * GRID, code, evaluations, pricing)
    except _CeilingReached:
        return SolveResult(term_months, None, BIND_EXH, evaluations)
    return SolveResult(term_months, None, BIND_AFF if stretches else BIND_CEIL, evaluations)


def _largest_affordable(probe, first: int, last: int, try_top_first: bool):
    """The last affordable position of a stretch on which affordability is a prefix, or None."""
    if try_top_first:  # the common case: the whole request is affordable
        pricing, affordable = probe(last)
        if affordable:
            return last, pricing
        if first == last:
            return None
    pricing, affordable = probe(first)
    if not affordable:
        return None
    if not try_top_first:
        if first == last:
            return first, pricing
        top_pricing, top_affordable = probe(last)
        if top_affordable:
            return last, top_pricing
    lo, lo_pricing, hi = first, pricing, last  # lo is affordable, hi is not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        pricing, affordable = probe(mid)
        if affordable:
            lo, lo_pricing = mid, pricing
        else:
            hi = mid
    return lo, lo_pricing


def exhaustive_search(
    evaluator: PriceEvaluator, term_months: int, risk_grade: int, applicant_age_years: float, is_joint: bool,
    amount_cap: float, requested_amount: float | None, max_affordable_instalment: float,
) -> float | None:
    """The same answer as `solve_term`, by pricing every R100 candidate. Slow: run it to check a new rate card.

        exhaustive_search(evaluator, 60, 9, 38.0, False, 500_000.0, None, 1_560.0)   # 50000.0
    """
    top = min(PRODUCT_MAX_AMOUNT, amount_cap, requested_amount if requested_amount and requested_amount > 0 else INF)
    best = None
    for position in range(_ceil_grid(PRODUCT_MIN_AMOUNT), _floor_grid(top) + 1):
        pricing = evaluator.evaluate(position * GRID, term_months, risk_grade, applicant_age_years, is_joint)
        if pricing.priced and pricing.instalment <= max_affordable_instalment:
            best = position * GRID
    return best
