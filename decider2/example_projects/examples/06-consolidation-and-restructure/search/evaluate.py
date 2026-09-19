"""Per-scenario evaluation: the record-tier pipeline applied to the candidate frame.

Everything in this file runs once per assessment, over a frame whose rows are
scenarios. Nothing here knows the search exists. That is the point: if this
pipeline could tell it was inside a search, it would be a second entry point
into affordability and it would drift from the first one (spec question 9).

The sequence, per spec 5.6.2:

    reduce the inventory -> re-derive obligations -> compute the required advance
    (circular solve) -> re-derive affordability -> price by product -> measure

Five of those six are library capabilities used unchanged. The only project-owned
arithmetic is the inventory reduction and the measurement.
"""

from decider2 import Branch, Loop, module, param, step
from decider2.frame import Aggregate, Join
from decider2.values import na

from credit_core import affordability, credit_life, fees, obligations, rounding
from products.bond_40.pricing import BondFurtherAdvance
from products.card_20.pricing import CardBalanceTransfer
from products.drive_30.pricing import DriveRefinance
from products.flex_11.pricing import FlexConsolidation


# --- 1. reduce the inventory --------------------------------------------------
#
# The scenario frame is 400 rows. The account frame is up to 80 rows. Their
# product - which accounts survive in which scenario - is at most 32 000 rows,
# which is a small polars join, and it is the natural shape for this work.
#
# THIS IS WHERE core.obligations' "one capability, two shapes of answer" problem
# (library 6.4) lands, and the library got the seam wrong. See FRAMEWORK-DEMANDS
# D15. Projects 03 and 07 want the scalar aggregate; this project wants the
# per-element annotation AND the aggregate, 400 times over. The right seam is not
# "one capability with two output shapes" but two modules sharing steps:
#
#   obligations.treat    record tier, one account -> obligation_treatment_code,
#                        treated_instalment. Reusable at any cardinality.
#   obligations.total    frame tier, group_by -> existing_obligations and friends.
#
# Project 03 composes both and discards the first's output. This project keeps it.
# Neither forks anything.

ReduceInventory = Join(
    "account_frame",
    on="client_id",
    how="cross_masked",  # membership by bit test against settlement_set_mask
    stable=True,
) | module(
    "is_settled_in_scenario",
    "surviving_balance",
    "surviving_limit",
    name="inventory_reduction",
)


@step(output="surviving_limit")
def surviving_limit(
    is_settled_in_scenario: bool,
    is_revolving: bool,
    credit_limit: float,
    closure_required: bool,
    limit_reduction_required: bool,
) -> float:
    """Post-action limit for a settled revolving account: closed, reduced, or intact.

    Spec 5.6.2 step 1. A settled card with its limit intact is a re-accumulation
    waiting to happen, and core.obligations must see the POST-ACTION state. Every
    lender that has built this has computed obligations on the pre-action state
    once, and discovered it when the client re-spent the card inside three months.

    The closure/reduction requirement is product policy (product 20 requires it,
    product 11 requires it above a limit threshold), so it arrives as a column
    from the routing table rather than being decided here.
    """
    pass  # 0 if closure_required, limit - transferred if reduction, else limit


ReDeriveObligations = ReduceInventory | obligations.Treat | Aggregate(
    by="scenario_id",
    metrics={
        "existing_obligations_after": "sum(treated_instalment)",
        "obligations_internal_after": "sum(treated_instalment where is_internal)",
        "obligations_external_after": "sum(treated_instalment where not is_internal)",
        "worst_arrears_months_after": "max(months_in_arrears)",
        "revolving_utilisation_after": "sum(surviving_balance) / sum(surviving_limit)",
        "accounts_after": "count()",
        "providers_after": "n_unique(provider_code)",
        "settled_instalment_total": "sum(instalment where is_settled_in_scenario)",
        "settled_remaining_cost_total": "sum(remaining_cost where is_settled_in_scenario)",
        "settled_weighted_rate": "weighted_mean(nominal_annual_rate, balance) where is_settled_in_scenario",
        "longest_settled_remaining_term": "max(remaining_term_months where is_settled_in_scenario)",
        "settlement_total": "sum(settlement_amount where is_settled_in_scenario)",
    },
    stable=True,
)


# --- 2. the circular solve ----------------------------------------------------
#
# Spec 5.6.2 step 3. Capitalised fees depend on the advance; the advance depends
# on the fees. The rate depends on the amount band; the amount depends on the
# rate through the capitalised credit life premium. On product 30 the amount also
# moves the LTV band, which moves the rate again, in a direction that may oppose
# the amount band's - so the solve has to survive TWO rate channels.
#
# This is doc 03 8.3's Loop used exactly as designed, and it is the one place in
# this sketch where the existing combinator needed nothing added except an
# output. See FRAMEWORK-DEMANDS D8.

SolveAdvance = Loop(
    "advance_not_converged",
    module(
        "required_advance",
        "amount_band",
        "ltv_band",
        "priced_rate",
        "capitalised_credit_life",
        "capitalised_fees",
        name="advance_solve_body",
    ),
    carries=["required_advance", "priced_rate", "capitalised_fees"],
    max_iterations=param(6, ge=1, le=12, description="Declared iteration bound."),
    tolerance=param(1.0, description="Rands. Convergence when the advance moves less."),
    # THE ADDITION doc 03 needs: a Loop that hits max_iterations currently just
    # stops. Here, not converging must be a RECORDED SCENARIO REJECTION rather
    # than a thrown error (spec 5.6.2), so the loop publishes whether it got
    # there and the rejection is raised by an ordinary intervention downstream.
    exhausted="carry",
    writes_exhausted_flag="solve_converged",
    writes_iteration_count="solve_iterations",
)


@step(output="required_advance")
def required_advance(
    settlement_total: float,
    settlement_buffer: float,
    new_money_requested: float,
    capitalised_fees: float,
    capitalised_credit_life: float,
    product_capitalised_costs: float,
) -> float:
    """Sum of settlements, buffer, new money, and everything capitalised into it.

    product_capitalised_costs is the arm's contribution: valuation and
    registration on product 30, conveyancing and deeds office on product 40, the
    transfer fee on product 20, nothing on product 11. It arrives as a column
    because the arm computed it, not as a branch inside this step.
    """
    pass  # straight sum; the circularity is in the Loop, not here


@step(output="settlement_buffer")
def settlement_buffer(
    settlement_total: float,
    buffer_pct: float = param(1.5, ge=0.0, le=5.0),
    buffer_cap: float = param(2500.0, ge=0.0, le=10000.0),
) -> float:
    """1.5% of the settlement total, capped at R2 500, added to the advance.

    Per-diem accrual means the true settlement amount is only knowable on the day
    the money arrives. Where the actual settlement is lower the residue goes to
    the new facility; where it is higher the shortfall is the client's, WHICH
    MUST BE DISCLOSED - so the buffer appears in the client-facing comparison and
    not only in the advance.
    """
    pass  # min(total * pct/100, cap)


# --- 3. affordability, re-derived, unforked -----------------------------------
#
# The library module, applied to the scenario frame. No wrapper, no copy, no
# second entry point. `.at()` rebinds the one input whose name genuinely differs
# (doc 03 5.2 layer 3) and nothing else moves.
#
# Note what is NOT here: income. There is no income derivation in this pipeline
# at all, because there is nothing in the plan that could vary it. Spec 5.6.1's
# requirement is met by the absence of code rather than by the presence of a
# guard, which is the strongest form the requirement can take.

AffordScenario = affordability.Assess.at(
    inputs={"existing_obligations": "existing_obligations_after",
            "instalment": "committed_monthly"}
)

# The restructure variant's twin. SAME MODULE, second instance, second params
# namespace, different input bindings. Doc 03 5.2 says "most projects never reach
# the third layer"; this project reaches it on page one and uses it for the
# single most important pair of numbers in the restructure record.
AffordScenarioStressed = affordability.Assess.at(
    inputs={"existing_obligations": "existing_obligations_after_stressed",
            "instalment": "committed_monthly",
            "net_monthly_income": "net_monthly_income_stressed",
            "living_expenses": "living_expenses_stressed"},
    name="affordability_stressed",
)


# --- 4. price it, by product --------------------------------------------------
#
# FOUR HETEROGENEOUS PRODUCTS, ONE FLOW, NO COPIES.
#
# Each row of the frame has exactly one product_code, so this is an ordinary
# n-way Branch (doc 03 8.2) and only the taken arm executes - which is where the
# short-circuiting advantage lives. The fan-out that produced several products
# per settlement set already happened, in the frame tier, in plan.py.
#
# `modifies` is the contract in contracts/product_offer.json. All four arms
# satisfy it. Adding product 21 is a fifth arm here, a routing table row, and a
# rate card - and nothing in the search, the objective, the interventions or the
# other four products moves (AC 15).
#
# The one thing doc 03 8.2 cannot express: product 20 has no term and no balloon,
# so `term_months` and `balloon_amount` are na() on that arm. Doc 03 requires
# every arm to produce every declared value with agreeing types, which would
# force a sentinel, which would put `term_months = 0` into the anti-harm rule.
# See FRAMEWORK-DEMANDS D4.

PriceByProduct = Branch(
    "product_code",
    {
        11: FlexConsolidation,
        20: CardBalanceTransfer,
        30: DriveRefinance,
        40: BondFurtherAdvance,
    },
    modifies=[
        "committed_monthly",
        "total_cost_of_credit",
        "horizon_months",
        "advance_or_limit",
        "nominal_annual_rate",
        "exposure_basis_code",
        "rate_basis_code",
        "rate_components",
        ("balloon_amount", "may_be_na"),
        ("term_months", "may_be_na"),
        "effective_annual_rate",
        "fee_breakdown",
        "rate_cell_id",
        "rate_card_version",
        "conditions_precedent",
        "product_rejection_codes",
    ],
    contract="contracts/product_offer.json",
    taps=["branch_path", "rate_cell_id"],
)


# --- 5. measure it ------------------------------------------------------------


@step(output="instalment_relief")
def instalment_relief(settled_instalment_total: float, committed_monthly: float) -> float:
    """Rands per month released: settled instalments less the new commitment."""
    pass  # settled_instalment_total - committed_monthly


@step(output="total_cost_delta")
def total_cost_delta(
    total_cost_of_credit: float,
    settled_remaining_cost_total: float,
) -> float:
    """New total cost of credit less the settled accounts' remaining cost.

    The number that decides whether the client was helped or harmed, and the
    number a consolidation flow optimising the instalment alone never computes.
    R120 000 of card debt at 22% over 36 months costs about R45 000 in interest;
    the same R120 000 against a home at 12% over 240 months costs about R197 000
    for a 71% instalment reduction. Both are true. Only one of them is on the
    screen unless this step exists.
    """
    pass  # total_cost_of_credit - settled_remaining_cost_total


@step(output="new_weighted_average_rate")
def new_weighted_average_rate(
    nominal_annual_rate: float,
    advance_or_limit: float,
    retained_weighted_rate: float,
    retained_balance: float,
) -> float:
    """The new facility's rate blended with what the client still carries."""
    pass  # balance-weighted mean over new facility and retained accounts


@step(output="bank_expected_value")
def bank_expected_value(
    nominal_annual_rate: float,
    advance_or_limit: float,
    horizon_months: int,
    probability_of_default: float,
    loss_given_default: float,
    margin_forgone_internal: float,
    cost_of_funds: float,
) -> float:
    """INCREMENTAL expected value, net of the margin forgone on the Bank's own book.

    `margin_forgone_internal` is the whole subtlety and it is an input, not an
    afterthought: a consolidation that refinances the Bank's own 26% loan at 19%
    DESTROYS value, and an objective measuring gross rather than incremental
    margin will happily recommend it. The measure in search/measures.py reads
    this output; this step is where the sign is got right.
    """
    pass  # (margin - cof) * exposure * life - PD*LGD*exposure - margin_forgone


Measures = module(
    instalment_relief,
    total_cost_delta,
    new_weighted_average_rate,
    bank_expected_value,
    name="scenario_measures",
    taps=["instalment_relief", "total_cost_delta"],
)


# --- the per-scenario pipeline ------------------------------------------------
#
# Written as a sequence because it IS a sequence: obligations before the solve,
# the solve before affordability, affordability before pricing is tested. Doc 03
# 8.1 - written order is execution order, visible in one place.
#
# fuse() on the solve body because it is the hot region: a six-iteration loop
# over 400 rows, arithmetic-dominated, with no cheap arms to short-circuit. It
# changes codegen only and cannot change the answer (doc 02 1.2), which is what
# makes it safe to put here rather than in a performance branch.

EvaluateScenario = (
    ReDeriveObligations
    | module(settlement_buffer, name="buffer")
    | fuse(SolveAdvance)
    | AffordScenario
    | PriceByProduct
    | Measures
)
