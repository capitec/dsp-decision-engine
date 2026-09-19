"""Product 11 - Flex Loan Consolidation. Unsecured, settlement-linked, the default route.

Owner: the Flex Loan product team. Release cadence: weekly.
Nothing in this file is imported by products 20, 30 or 40, and nothing in this
file imports them. That is the test of whether "four subflows in one flow" is
real or whether it is four copies with a shared header.

What this arm shares with the other three: contracts/product_offer.json and the
credit-core capabilities. That is all.
"""

from decider2 import module, param, step
from decider2.values import missing_as, na
from decider2.params import overlayable

from credit_core import credit_life, fees, instalment as core_instalment, rate_card, rounding


# --- rate: an ABSOLUTE rate, plus a consolidation-specific margin grid ---------


@step(output="rate_cell")
def rate_cell(
    required_advance: float,
    term_months: int,
    risk_grade: int,
    decision_date: "date",
    tables,
) -> "Row":
    """Look up the cell. Returns a ROW, not a scalar.

    `.nominal_annual_rate`, `.cell_id`, `.card_version`, `.band_edge`.

    Doc 03's provisional Table sketch returns `tables.term.max_loan[term]` - a
    scalar - and this project falsifies that assumption four different ways.
    Even here, the simplest of the four, a scalar return loses `cell_id`, which
    is the thing library 8 property 1 requires and the thing an auditor asks for.
    See FRAMEWORK-DEMANDS D5.
    """
    pass  # tables.rate_card_11.at(amount_band=..., term_months=..., risk_grade=...)


@step(output="external_proportion_margin_bps")
def external_proportion_margin_bps(
    settlement_total_external: float,
    settlement_total: float,
    risk_grade: int,
    tables,
) -> float:
    """4 external-proportion bands x 12 grades = 48 cells.

    Refinancing the Bank's own book at a consolidation rate is not the same
    transaction as taking a client's debt off a competitor, and this grid is
    where that shows up in the price. It is also why OBJ-04 subtracts the margin
    forgone on internal accounts: the same economic fact, once in the price and
    once in the objective, and they must not disagree about the sign.
    """
    pass  # tables.consol_margin_11.at(external_band=..., risk_grade=...)


@step(output="nominal_annual_rate")
def nominal_annual_rate(
    rate_cell: "Row",
    external_proportion_margin_bps: float,
    rate_add_on_bps: float = overlayable(0.0, scope=["product_code", "channel_code"]),
) -> float:
    """Card rate + external-proportion margin + any overlay add-on.

    `overlayable(...)` marks this field as an overlay TARGET and declares the
    scope key the overlay register resolves against. Three things follow, and
    together they are FRAMEWORK-DEMANDS D3:

      - `params.rate_add_on_bps` is the resolved value: 75.0 in the branch
        channel this quarter (ADJ-RATE-002).
      - `params.base.rate_add_on_bps` is 0.0, and is recorded beside it.
      - Running the whole assessment with the stack disabled is the SAME
        compiled kernel over `resolved.base()`. No second implementation, which
        is acceptance criterion 10 and library question 12.

    The alternative - reading an overlay object inside the step - cannot work:
    an njit'd step has no access to Python runtime state, so anything a compiled
    step reads must arrive as an argument (doc 03 4.2). Resolving overlays into
    params before the kernel runs is the only shape that survives compilation,
    and it is also the shape that makes the stack-disabled run free.
    """
    pass  # cell rate + margin/100 + add_on/100


# --- the product's own policy set ---------------------------------------------
#
# Every one of these is this product team's to change on a Wednesday without
# touching the other three and without a Credit Risk Policy cycle (AC 9). They
# live in the `flex_11` params namespace, which is what bounds the blast radius
# to this module by construction (doc 03 4.1).


@step(output="product_rejection_codes")
def flex_policy_gates(
    accounts_settled: int,
    external_proportion_pct: float,
    required_advance: float,
    instalment_relief_pct: float,
    term_months: int,
    longest_settled_remaining_term: int,
    min_accounts: int = param(2, ge=2, le=8),
    min_external_pct: float = param(60.0, ge=0.0, le=100.0),
    min_amount: float = param(10000.0, ge=1000.0, le=50000.0),
    max_amount: float = param(500000.0, ge=50000.0, le=1000000.0),
    min_relief_pct: float = param(10.0, ge=0.0, le=30.0),
    max_term_over_longest: int = param(24, ge=0, le=60),
) -> list:
    """The arm's OWN rejections, distinct from the fourteen policy interventions.

    Fewer than 2 accounts; external proportion below 60% (so the product is not a
    disguised cash loan); advance outside R10 000 - R500 000; instalment relief
    below the floor; term beyond the lesser of 84 months and the longest settled
    remaining term + 24.

    Product rejections and policy interventions are kept apart deliberately.
    "Your advance was below our minimum" is a product fact a product team can
    change on Wednesday. "The total cost exceeded the anti-harm ceiling" is a
    credit policy decision with a Credit Committee behind it. A contact centre
    agent reading the rejection list needs to know which kind they are looking at,
    because only one of them is escalatable.
    """
    pass  # list of RJ-11-xx codes, ALL that apply, never just the first


@step(output="committed_monthly")
def flex_committed_monthly(
    required_advance: float,
    nominal_annual_rate: float,
    term_months: int,
    monthly_service_fee: float,
    credit_life_premium: float,
) -> float:
    """The contractual instalment including fees and premium. Standard treatment.

    Product 11 is the only one of the four whose committed_monthly is simply the
    contractual instalment. The other three each have a reason it is not, and
    those three reasons are the substance of the heterogeneity.
    """
    pass  # core.instalment over advance, rate, term + fees + premium


@step(output="total_cost_of_credit")
def flex_total_cost(committed_monthly: float, term_months: int) -> float:
    """Every rand the client pays. No balloon, no reversion, no sub-term. """
    pass  # committed_monthly * term_months


FlexConsolidation = module(
    rate_cell,
    external_proportion_margin_bps,
    nominal_annual_rate,
    flex_policy_gates,
    flex_committed_monthly,
    flex_total_cost,
    name="flex_11",
    params="config/products/flex_11.json",
    contract="contracts/product_offer.json",
    taps=["rate_cell_id", "external_proportion_margin_bps"],
)

# The consolidation-specific scorecard is a separate module composed ahead of
# this one in the pipeline, not a step inside it. It has characteristics plain
# granting does not have - the count of accounts being settled, the proportion of
# income currently servicing debt, the number of providers exited, whether the
# client has consolidated before - and three of those four VARY PER SCENARIO.
#
# That is worth stating plainly because it is easy to miss: the risk grade is not
# an assessment-level invariant on this product. It is a per-scenario value, it
# moves the rate cell, it moves the price, and it therefore sits INSIDE the
# circular solve rather than before it. A grade computed once per assessment
# would be wrong for 399 of the 400 scenarios.
