"""Product 30 - Drive Finance refinance. Secured against a vehicle.

Owner: the Drive Finance product team.

Structurally the most different of the four, because the collateral has to be
valued, its condition affects the price, and the security has to move from one
provider to another while the client keeps driving the car.

The interesting engineering is the double rate channel: LOAN-TO-VALUE IS A KEY
DIMENSION OF THE RATE CARD, so the price depends on the amount through the amount
band AND through the LTV band, and those two can move the rate in opposite
directions. The circular solve in search/evaluate.py has to survive both.
"""

from decider2 import module, param, step
from decider2.values import Maybe, missing_as, na, fresh_until
from decider2.params import overlayable


# --- valuation: 60 000 rows, then two adjustment tables ----------------------


@step(output="guide_value")
def guide_value(
    vehicle_make_model_year_code: int,
    decision_date: "date",
    tables,
) -> "Row":
    """Trade guide lookup, ~60 000 make/model/year/derivative rows x 6 values.

    `fresh_until` on the table itself, not on this step: the guide's current
    month's edition is REQUIRED and a stale edition is a hard stop for product 30
    (spec 4.8). The hard stop withdraws the product from routing and records the
    withdrawal - it does not fail the assessment, and the client is told why.
    """
    pass  # tables.vehicle_guide.at(mmy_code=...), returns retail/trade/new values


@step(output="mileage_adjustment")
def mileage_adjustment(
    vehicle_mileage_km: int,
    vehicle_age_years: float,
    tables,
) -> float:
    """14 kilometre bands x 8 age bands = 112 cells."""
    pass  # tables.mileage_adj.at(km_band=..., age_band=...)


@step(output="condition_haircut")
def condition_haircut(
    vehicle_condition_code: int = missing_as(99),  # 99 = worst band, deliberately
    vehicle_age_years: float = 0.0,
    tables=None,
) -> float:
    """5 conditions x 8 age bands = 40 cells.

    A NULL CONDITION CODE FORCES THE WORST BAND. That is a deliberate incentive
    to inspect, and `missing_as(99)` is exactly the right shape for it: the
    policy is declared in the signature where a reviewer looks for the interface,
    and the step body sees a plain integer with nothing to forget (doc 03 1,
    tier 2). Writing this as `if code is None: code = 99` inside the body would
    put a credit policy decision in a place no reviewer reads.
    """
    pass  # tables.condition_haircut.at(condition=..., age_band=...)


@step(output="adjusted_retail_value")
def adjusted_retail_value(
    guide_value: "Row",
    mileage_adjustment: float,
    condition_haircut: float,
) -> float:
    """Guide retail, adjusted for mileage and condition. The LTV denominator."""
    pass  # guide.retail * (1 + mileage_adj) * (1 - haircut)


# --- LTV: a gate, a rate key, and an overlay target --------------------------


@step(output="ltv_pct")
def ltv_pct(
    required_advance: float,
    balloon_amount: float,
    adjusted_retail_value: float,
) -> float:
    """(advance + balloon) / adjusted retail value.

    THE BALLOON IS IN THE NUMERATOR. A balloon lowers the instalment and is
    therefore attractive to the objective; leaving it out of LTV would let a
    structure that raises the Bank's exposure pass an exposure gate.
    """
    pass  # (required_advance + balloon_amount) / adjusted_retail_value * 100


@step(output="ltv_cap_pct")
def ltv_cap_pct(
    vehicle_age_years: float,
    tables,
    ltv_cap_override: float = overlayable(0.0, scope=["product_code", "vehicle_age_band"]),
) -> float:
    """<=3 years 110%, 4-6 years 100%, 7-9 years 85%, >=10 years ineligible.

    ADJ-LTV-004 tightens the 4-6 band from 100% to 90% "pending a loss review",
    applied in October 2025, review date March 2026, AND IT IS STILL IN FORCE.
    That is change scenario 3, and it is in the register as a live exception
    rather than as a story, because `review_date` is a required field on every
    overlay and an overlay past it must surface.

    The overlay does not touch the rate card. It reduces a cap WITHOUT REISSUING
    THE CARD, which is the whole reason the overlay mechanism exists - and it is
    also why `params.base.ltv_cap_pct` must remain readable, so that "what would
    we have done without the overlay" is answerable for tens of thousands of
    clients on the day somebody asks what unwinding it would do.
    """
    pass  # table lookup by age band, then apply the overlay


@step(output="rate_cell")
def drive_rate_cell(
    required_advance: float,
    term_months: int,
    risk_grade: int,
    ltv_pct: float,
    decision_date: "date",
    tables,
) -> "Row":
    """60 amount bands x 61 terms x 12 grades x 5 LTV bands = 219 600 cells.

    Both `required_advance` and `ltv_pct` are carried values of the solve, and
    both key this lookup. Six iterations, R1 tolerance. Non-convergence is a
    RECORDED SCENARIO REJECTION (RJ-SOLVE-01), not a thrown error - which matters
    because a thrown error in one of 400 candidate rows would abort the whole
    kernel and lose 399 valid evaluations.
    """
    pass  # tables.rate_card_30.at(amount_band, term, grade, ltv_band)


@step(output="vehicle_age_adjustment_bps")
def vehicle_age_adjustment_bps(vehicle_age_years: float, ltv_pct: float, tables) -> float:
    """Refinance-specific: 8 age bands x 5 LTV bands = 40 cells."""
    pass  # tables.rate_card_30_age_adj.at(age_band=..., ltv_band=...)


@step(output="nominal_annual_rate")
def drive_nominal_rate(
    rate_cell: "Row",
    vehicle_age_adjustment_bps: float,
    ltv_band_add_on_bps: float = overlayable(0.0, scope=["product_code", "ltv_band"]),
) -> float:
    """Card rate + refinance age adjustment + any LTV-band overlay add-on."""
    pass  # cell rate + (age_adj + add_on) / 100


# --- balloon, and why it is in total cost -------------------------------------


@step(output="balloon_amount")
def balloon_amount(
    required_advance: float,
    term_months: int,
    balloon_pct_requested: float = missing_as(0.0),
    max_pct_to_48m: float = param(30.0, ge=0.0, le=40.0),
    max_pct_at_60m: float = param(20.0, ge=0.0, le=30.0),
) -> Maybe[float]:
    """30% to 48 months, 20% at 60, 0% above. na() where the term forbids one."""
    pass  # min(requested, cap for this term), na() above 60 months


@step(output="total_cost_of_credit")
def drive_total_cost(
    committed_monthly: float,
    term_months: int,
    balloon_amount: Maybe[float],
) -> float:
    """Instalments over the term PLUS THE BALLOON.

    Spec 5.6.5: "the total cost of credit used by the anti-harm rule must include
    the balloon, or the rule can be defeated by structure rather than by
    argument."

    Read that twice. It describes an exploit: raise the balloon, drop the
    instalment, pass the anti-harm rule, and hand the client a R180 000 bullet
    payment in five years. Nothing about the exploit is illegal or even unusual,
    and an objective weighting instalment relief at 0.6 will find it
    automatically. The defence is one addition in this step.
    """
    pass  # committed_monthly * term_months + (balloon or 0)


@step(output="committed_monthly")
def drive_committed_monthly(
    required_advance: float,
    nominal_annual_rate: float,
    term_months: int,
    balloon_amount: Maybe[float],
    monthly_service_fee: float,
    credit_life_premium: float,
    insurance_premium_estimate: float,
) -> float:
    """Contractual instalment PLUS a required comprehensive insurance premium.

    Omitting the insurance premium overstates affordability by R700 to R1 400 a
    month on a typical vehicle. An uninsured vehicle is not security, so the
    premium is a condition precedent AND an affordability input, and both facts
    have to be true in the same record or the reckless lending review finds the
    gap.

    `insurance_premium_estimate` is zero where the client already holds cover, so
    the same step serves both cases and the estimate that was used is recorded.
    """
    pass  # amortisation with balloon + fees + premium + insurance


@step(output="product_rejection_codes")
def drive_policy_gates(
    vehicle_identified: bool,
    guide_value_available: bool,
    ltv_pct: float,
    ltv_cap_pct: float,
    vehicle_age_years: float,
    vehicle_mileage_km: int,
    implied_annual_mileage: float,
    existing_finance_settleable: bool,
    insurance_accepted: bool,
    term_months: int,
    months_since_original_registration: int,
    max_age_at_end_of_term: float = param(12.0, ge=6.0, le=15.0),
    max_mileage_km: int = param(220000, ge=100000, le=400000),
    max_annual_km: int = param(40000, ge=20000, le=80000),
    max_total_financed_life_months: int = param(84, ge=60, le=120),
) -> list:
    """Vehicle not identifiable; valuation unavailable; LTV above cap; age or
    mileage gate; existing finance account not settleable; insurance refused; a
    refinance extending total financed life beyond 84 months from original
    registration.
    """
    pass  # list of RJ-30-xx codes, all that apply


DriveRefinance = module(
    guide_value,
    mileage_adjustment,
    condition_haircut,
    adjusted_retail_value,
    ltv_pct,
    ltv_cap_pct,
    drive_rate_cell,
    vehicle_age_adjustment_bps,
    drive_nominal_rate,
    balloon_amount,
    drive_committed_monthly,
    drive_total_cost,
    drive_policy_gates,
    name="drive_30",
    params="config/products/drive_30.json",
    contract="contracts/product_offer.json",
    taps=["ltv_pct", "ltv_cap_pct", "adjusted_retail_value", "rate_cell_id", "condition_haircut"],
)

# ltv_pct and ltv_cap_pct are BOTH tapped, not just the verdict. "Your LTV was
# 97.2% against a cap of 90%" is an answer a client can act on - they can put
# down a deposit. "LTV above cap" is not. Spec 5.7.2 demands actual and threshold
# for policy interventions; the same courtesy applied to product gates costs two
# int64 columns and removes a whole class of contact centre call.
