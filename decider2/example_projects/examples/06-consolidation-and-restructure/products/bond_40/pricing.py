"""Product 40 - Home Loan Further Advance. The one most likely to end up at the ombud.

Owner: the Home Loan product team. Warning wording: Regulatory Compliance.

The most powerful instalment reduction available and the most dangerous. Three
mechanisms in this file exist solely to stop the objective choosing it for the
wrong reason:

  the MANDATORY 84-MONTH SUB-TERM, so total cost is measured over something
  comparable rather than over 240 months;
  the WARNING ACKNOWLEDGEMENT, enforced in ROUTING so an unacknowledged scenario
  is never generated;
  a MARGIN-OVER-REFERENCE rate representation, recorded as two components so a
  replay survives the reference rate moving twice.

R120 000 of card debt at 22% over 36 months costs about R45 000 in interest. The
same R120 000 against a home at 12% over 240 months costs about R197 000 - a
337% increase - for a 71% instalment reduction. The instalment reduction is real,
the client will want it, and it is still the wrong answer for most clients.
"""

from decider2 import module, param, step
from decider2.values import Maybe, fresh_until, missing_as, na
from decider2.params import overlayable


# --- valuation and equity ------------------------------------------------------


@step(output="property_value")
def property_value(
    property_valuation_amount: float = fresh_until("valuation.date", within_months=24),
    property_avm_amount: float = fresh_until("avm.date", within_days=90, on_stale="degrade"),
    required_advance: float = 0.0,
    avm_sufficient_below: float = param(300000.0, ge=0.0, le=1000000.0),
) -> float:
    """AVM is sufficient where the last physical valuation is under 24 months old
    AND the advance is below R300 000. Otherwise a physical valuation is a
    CONDITION PRECEDENT, adding 10 to 15 working days.

    Two freshness declarations with two different stale behaviours in one
    signature, which is the case doc 03 1's null policy does not cover: staleness
    is not nullness, the two have different remedies, and the remedy here depends
    on ANOTHER input's value. `on_stale="degrade"` on the AVM means fall back to
    the physical valuation requirement; the physical valuation's own window is a
    hard bound on what it can support. See FRAMEWORK-DEMANDS D12.
    """
    pass  # AVM where permitted, else physical valuation, else condition precedent


@step(output="available_equity")
def available_equity(
    property_value: float,
    ltv_cap_pct: float,
    bond_registered_amount: float,
    outstanding_home_loan_balance: float,
    prior_encumbrances: float = missing_as(0.0),
) -> float:
    """min(value x LTV cap, REGISTERED BOND AMOUNT) - balance - prior encumbrances.

    The registered bond amount caps a further advance without a new registration.
    Where the advance exceeds it, A NEW REGISTRATION IS REQUIRED, which changes
    the cost (conveyancing tariff, deeds office, attorney) and the timeline
    (6-10 weeks) materially - and both of those feed back into the circular
    solve, because the registration cost is capitalised.

    Prior encumbrances - second bonds, builders' liens - come from the deeds feed
    and are missing often enough that `missing_as(0.0)` is the WRONG default and
    is used here anyway, deliberately, with a recorded flag: a missing deeds feed
    is a data-quality condition on the outcome, not a licence to assume zero. The
    flag is what makes the assumption visible rather than the default hiding it.
    """
    pass  # min(value * cap/100, registered) - balance - encumbrances


@step(output="ltv_cap_pct")
def bond_ltv_cap(
    occupancy_type_code: int,
    bond_age_months: int,
    owner_occupied_cap: float = param(90.0, ge=50.0, le=100.0),
    investment_cap: float = param(75.0, ge=40.0, le=90.0),
    new_bond_cap: float = param(80.0, ge=40.0, le=95.0),
    new_bond_months: int = param(12, ge=0, le=36),
) -> float:
    """90% owner-occupied, 75% investment, 80% where the bond is under 12 months old.

    The binding cap is the lowest applicable, and WHICH ONE BOUND is recorded.
    Three caps that can all apply and a record that says only "LTV cap 80%" leaves
    the client unable to know whether waiting three months changes anything.
    """
    pass  # min of applicable caps, with the binding one tapped


# --- the third rate representation ---------------------------------------------


@step(output="margin_cell")
def margin_cell(
    risk_grade: int,
    ltv_pct: float,
    term_months: int,
    occupancy_type_code: int,
    decision_date: "date",
    tables,
) -> "Row":
    """12 grades x 8 LTV bands x 6 term bands x 2 occupancy types = 1 152 cells.

    Returns `.margin_bps`, NOT a rate. A fourth rate card with a fourth key set
    and a third representation, in one flow, all four live at once. This is what
    spec question 6 is really asking, and the answer is that the lookup returns a
    row of the card and the MEANING of what it returns is product logic.
    """
    pass  # tables.rate_card_40.at(grade, ltv_band, term_band, occupancy)


@step(output="nominal_annual_rate")
def bond_nominal_rate(
    margin_cell: "Row",
    shared,
    pd_multiplier: float = overlayable(1.0, scope=["product_code", "segment_code"]),
) -> float:
    """reference rate + margin. Both components recorded; the sum alone is unreplayable.

    `shared.reference_rate` (doc 03 4.2) because it is genuinely global and must
    not be copied into four product bundles. A record showing only 12.35% cannot
    be replayed once the reference rate has moved twice, and it will have moved
    twice before the ombud arrives.

    ADJ-PD-007 multiplies the PD by 1.25, which moves the GRADE, which moves the
    MARGIN, which moves which scenarios pass CON-INT-06. An overlay two
    derivations upstream of the thing it changes. The record must carry the
    unadjusted PD, the unadjusted grade and the unadjusted margin or the cascade
    is unexplainable - which is three `params.base` reads and no new mechanism.
    """
    pass  # shared.reference_rate + margin_cell.margin_bps / 100


# --- the mandatory sub-term, which is the whole defence ------------------------


@step(output="horizon_months")
def bond_horizon(
    bond_remaining_term_months: int,
    applicant_age_years: float,
    sub_term_max: int = param(84, ge=36, le=120),
    max_age_at_end: float = param(70.0, ge=60.0, le=80.0),
) -> int:
    """84, not 240. THE CONSOLIDATED PORTION'S CONTRACTUAL AMORTISATION.

    The bond's own term is the lesser of 240 months, the remaining bond term, and
    the term ending at the client's 70th birthday. The CONSOLIDATED PORTION is
    amortised over at most 84 months, by way of a required additional payment
    above the bond instalment, and the anti-harm rule is evaluated against that
    structure.

    This one number is what makes product 40 defensible. Against 240 months the
    anti-harm threshold of 15% is unreachable by construction, product 40 wins
    every scenario it can carry, and the flow becomes a machine for securing
    unsecured debt against homes. Against 84 months it competes honestly with
    product 11 and usually loses, which is the correct outcome.
    """
    pass  # min(sub_term_max, bond term, term to age cap)


@step(output="committed_monthly")
def bond_committed_monthly(
    required_advance: float,
    nominal_annual_rate: float,
    horizon_months: int,
    bond_instalment_existing: float,
) -> float:
    """Bond instalment PLUS the required additional payment on the consolidated portion.

    Publishing only the incremental bond instalment - which is what a client asks
    for and what a naive implementation computes - would make product 40 look
    like a R1 321 answer when the client is committed to R2 470. The contract's
    composition-time assertion catches this: affordability binds to this column.
    """
    pass  # amortisation of the advance over horizon_months, at the blended rate


@step(output="total_cost_of_credit")
def bond_total_cost(committed_monthly: float, horizon_months: int) -> float:
    """Over 84 months, matching the horizon. Comparable to products 11 and 30."""
    pass  # committed_monthly * horizon_months


@step(output="product_capitalised_costs")
def bond_capitalised_costs(
    required_advance: float,
    bond_registered_amount: float,
    new_registration_required: bool,
    tables,
) -> float:
    """Conveyancing tariff (~24 amount bands x 3 components), valuation, deeds office.

    Capitalised, so it feeds the circular solve: the cost depends on the advance
    and the advance depends on the cost. And it STEPS: crossing the registered
    bond amount adds a new registration, which adds R12 000-R24 000 at once. A
    solve over a discontinuous function needs its iteration bound respected and
    its non-convergence recorded, and this is the step that makes the function
    discontinuous.
    """
    pass  # tariff lookup + valuation fee + (registration block if required)


@step(output="product_rejection_codes")
def bond_policy_gates(
    client_holds_bond: bool,
    bond_in_arrears: bool,
    bond_in_legal_process: bool,
    available_equity: float,
    required_advance: float,
    ltv_pct: float,
    ltv_cap_pct: float,
    valuation_available: bool,
    valuation_stale: bool,
    horizon_months: int,
    sub_term_affordable: bool,
    warning_acknowledged: bool,
) -> list:
    """No bond, or bond in arrears; insufficient equity; LTV above cap; valuation
    unavailable or stale; term beyond the age limit; warning not acknowledged;
    SUB-TERM STRUCTURE UNAFFORDABLE.

    The last one is the common rejection and it is the correct one. A client for
    whom the 240-month instalment is affordable and the 84-month sub-term is not
    has been told, correctly, that the product does not work for them - rather
    than being sold the 240-month version.
    """
    pass  # list of RJ-40-xx codes, all that apply


@step(output="conditions_precedent")
def bond_conditions(
    new_registration_required: bool,
    physical_valuation_required: bool,
    warning_version: str,
) -> list:
    """Registration, valuation, and the warning acknowledgement WITH ITS VERSION.

    Change scenario 11: "the further advance warning wording changes; decisions
    before the change keep the old wording and version in evidence forever."
    The version is captured into the record at decision time, not looked up at
    read time, which is the only shape that survives the wording changing.
    """
    pass  # list of CP-40-xx with owners and expected durations


BondFurtherAdvance = module(
    property_value,
    available_equity,
    bond_ltv_cap,
    margin_cell,
    bond_nominal_rate,
    bond_horizon,
    bond_committed_monthly,
    bond_total_cost,
    bond_capitalised_costs,
    bond_policy_gates,
    bond_conditions,
    name="bond_40",
    params="config/products/bond_40.json",
    contract="contracts/product_offer.json",
    taps=["margin_cell_id", "ltv_pct", "ltv_cap_pct", "available_equity", "horizon_months"],
)

# Every product 40 scenario is CONDITIONAL and always requires re-derivation:
# registration takes 6 to 10 weeks and every settlement quotation in the set will
# expire before disbursement. output/execution_package.py is written on that
# assumption. Spec 5.6.7: "re-derivation is not an edge case; it is the normal
# path, and it must be designed for rather than handled."
