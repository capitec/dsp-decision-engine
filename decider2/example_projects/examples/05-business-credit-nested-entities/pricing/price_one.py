"""Stage 5.12b -- price ONE candidate. Candidate grain.

Exactly the same authoring shape as `entities/adverse/classify.py`: pure scalar
steps over one record, where "one record" is one (amount, term) pair. The
circularity s5.12 describes -- the rate depends on the security type which
depends on the amount which depends on the rate -- is gone, because at this
grain the amount is a coordinate and not an unknown.

Every step here is evaluated 200-600 times per application, and the module is
wrapped in `parallel(...)` in the pipeline because this is the one uniform body
in the flow.
"""

from decider2 import module, param, step, table, overlay, round_half_up
from grains import Candidate

RATE_CARD = table(
    "business_rate_card",
    key=("amount_band_index", "term_months", "risk_grade", "security_type"),
    columns=("base_rate",),
    source="tables/rate_card.csv",         # 40 x 55 x 12 x 4 = 105 600 cells
    effective_dated=True,
    owner="Treasury",
    cadence="monthly",
    overlay=overlay.position(11, scopes=("rate_cell_range", "risk_grade",
                                         "security_type")),
    emits_cell_id=True,                    # s10 acceptance 14: every table read
                                           # records its version AND the cell read
)


# -- 1. cap the amount ------------------------------------------------------

def candidate_amount(
    band_ceiling: float,
    requested_amount: float,
    appetite_maximum: float,
    security_cover_capacity: float,
    group_exposure_headroom: float,
    product_maximum_amount: float,
    statement_based_cap: float,
    total_surety_cover: float,
    is_surety_backed: bool,
) -> float:
    """The least of the eight ceilings in s5.12 step 1, at this band."""
    pass


def binding_ceiling_code(
    band_ceiling: float, requested_amount: float, appetite_maximum: float,
    security_cover_capacity: float, group_exposure_headroom: float,
    product_maximum_amount: float, statement_based_cap: float,
    total_surety_cover: float,
) -> int:
    """WHICH of the eight bound. Not derivable afterwards from the amount alone
    when two ceilings tie, so it is computed here and carried."""
    pass


# -- 2. security type, which depends on the amount --------------------------

def cover_ratio(
    adjusted_collateral_value: float,
    total_surety_cover: float,
    total_guarantee_cover: float,
    candidate_amount: float,
) -> float:
    """Total adjusted cover divided by THIS candidate's amount."""
    pass


def security_type(
    cover_ratio: float,
    total_surety_cover: float,
    candidate_amount: float,
    fully_secured_ratio: float = param(1.00, ge=0.0, le=2.0),
    partially_secured_ratio: float = param(0.50, ge=0.0, le=2.0),
    surety_backed_ratio: float = param(0.25, ge=0.0, le=2.0),
) -> int:
    """1 fully secured / 2 partially / 3 surety-backed / 4 unsecured.

    "Security type depends on the offered amount, and the rate depends on the
    security type, and the affordable amount depends on the rate." At this grain
    that sentence describes three columns of one row.

    Change scenario 9 adds a fifth security type and takes the rate card to
    131 400 cells. Here that is one more return value and one more table
    dimension -- a table change plus a step change, no re-plumbing, because the
    security type was never an implicit index into anything.
    """
    pass


# -- 3. the rate ------------------------------------------------------------

def base_rate(
    amount_band_index: int, term_months: float,
    risk_grade: int, security_type: int, decision_date: int,
) -> float:
    """105 600-cell lookup. Emits `rate_cell_id` and `rate_card_version`."""
    pass


def sector_premium(sector_code: int) -> float:
    """0-150 bps from the sector table."""
    pass


def relationship_discount(
    relationship_tier_code: int,
    authority_level_code: int,
    max_discount_bps: float = param(75.0, ge=0.0, le=300.0),
) -> float:
    """0-75 bps, authority-limited."""
    pass


def rate_floor(
    cost_of_funds: float, capital_charge: float,
    probability_of_default: float, loss_given_default: float,
    operating_cost: float,
) -> float:
    """Cost of funds + capital charge + expected loss at the grade's PD +
    operating cost."""
    pass


def nominal_annual_rate(
    base_rate: float, sector_premium: float,
    relationship_discount: float, rate_floor: float,
) -> float:
    """The composed rate. Overlay position 11 -- Treasury's mid-month basis-point
    add-on over a declared cell range -- applies between the premium and the
    discount, in the declared stack order.

    s5.12: "The rate record carries the cell identifier, the cell's own value,
    every add-on with its overlay identifier, and the final rate, so that a
    repricing overlay is never mistaken for a rate card change."
    """
    pass


def rate_floor_binds(nominal_annual_rate: float, rate_floor: float) -> bool:
    """Recorded as the binding price constraint where it bites."""
    pass


# -- 4-6. fees, instalment, coverage ----------------------------------------

def initiation_fee(candidate_amount: float, regulatory_regime_code: int,
                   product_code: int, decision_date: int) -> int:
    """core.fees under the regulated regime, the business schedule otherwise.
    Money is a scaled int64 of cents (doc 03 s1.2)."""
    pass


def instalment(
    candidate_amount: float, term_months: float, nominal_annual_rate: float,
    initiation_fee: int, monthly_service_fee: int, assessment_fee: int,
    capitalise_costs: bool,
) -> int:
    """core.instalment. Called once per candidate -- 200-600 times per
    application -- which is s00 s6.6's "called inside iterative solves tens of
    times per application". Here it is not inside a solve; it is a column."""
    pass


def candidate_dscr(
    ebitda_after_haircuts: float, tax_paid: float, maintenance_capex: float,
    subordinated_director_loan_movement: float,
    existing_interest: float, existing_scheduled_principal: float,
    instalment: int,
) -> float:
    """Debt service coverage at THIS candidate's instalment.

    The same computation as financial/measures.py's `debt_service_coverage`,
    re-pinned to the Candidate grain. One definition, two grains -- see
    FRAMEWORK-DEMANDS D08. Duplicating it would be the obvious thing and would
    guarantee FV-02 eventually failing.
    """
    pass


def meets_coverage(
    candidate_dscr: float, interest_cover: float, post_facility_gearing: float,
    risk_grade: int, segment_code: int,
) -> bool:
    """The three thresholds from the grade-band table."""
    pass


def owner_affordability_passes(
    regulatory_regime_code: int, owner_is_income_source: bool,
    instalment: int, business_net_profit_after_drawings: float,
    statutory_affordability_verdict: int,
) -> bool:
    """s5.12 step 7. Under the regulated regime this is project 02's statutory
    assessment on the natural person and a fail is HARD; unregulated, drawings
    are deducted from EBITDA before the coverage test instead."""
    pass


# -- 7-8. admissibility and the binding constraint --------------------------

def is_admissible(
    candidate_amount: float, meets_coverage: bool,
    owner_affordability_passes: bool, product_minimum_amount: float,
) -> bool:
    """The candidate could be offered."""
    pass


def binding_constraint_code(
    binding_ceiling_code: int, meets_coverage: bool, candidate_dscr: float,
    interest_cover: float, post_facility_gearing: float,
    owner_affordability_passes: bool, rate_floor_binds: bool,
) -> int:
    """One of twelve codes. Recorded for EVERY candidate, admissible or not --
    s5.12's "Records: every candidate evaluated, in order, with its rate cell,
    instalment, DSCR and rejection reason. A client asking 'why not
    R2 000 000?' gets the row for R2 000 000."
    """
    pass


PriceCandidate = module(
    candidate_amount, binding_ceiling_code,
    cover_ratio, security_type,
    base_rate, sector_premium, relationship_discount, rate_floor,
    nominal_annual_rate, rate_floor_binds,
    initiation_fee, instalment, candidate_dscr, meets_coverage,
    owner_affordability_passes, is_admissible, binding_constraint_code,
    name="price_candidate",
    grain=Candidate,
    taps=["rate_cell_id", "rate_card_version", "base_rate",
          "nominal_annual_rate", "security_type", "candidate_dscr",
          "is_admissible", "binding_constraint_code"],
    contract="contracts/price_candidate.json",
)
