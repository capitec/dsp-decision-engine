"""Product 20 - Everyday Card balance transfer. The odd one out.

Owner: the Card product team.

Revolving. NO TERM AND NO INSTALMENT, and both the objective and the anti-harm
rule are defined in terms of things it does not have. This file is where na()
earns its place and where the rate card stops returning a scalar.

Three inversions to hold onto:

  the output is a LIMIT, not an amount
  the rate is TWO RATES AND A DATE, not a rate
  the affordability figure is a STRESSED payment, not the payment
"""

from decider2 import module, param, step
from decider2.values import Maybe, missing_as, na
from decider2.params import overlayable


# --- the rate card that breaks the scalar assumption --------------------------


@step(output="promo_cell")
def promo_cell(
    approved_limit: float,
    risk_grade: int,
    promotional_months: int,
    decision_date: "date",
    tables,
) -> "Row":
    """12 grades x 8 limit bands x 5 promotional durations = 480 cells.

    Returns `.promotional_rate`, `.cell_id`. There is no sensible scalar for a
    lookup to return here: the promotional rate alone is a mis-selling machine,
    and the promotional rate plus duration is still not a price.
    """
    pass  # tables.rate_card_20_promo.at(risk_grade=..., limit_band=..., promotional_months=...)


@step(output="reversion_cell")
def reversion_cell(approved_limit: float, risk_grade: int, tables) -> "Row":
    """12 grades x 20 limit bands = 240 cells. THE RATE THAT DECIDES THE TOTAL COST.

    A second lookup against a second card with a different key set, resolved in
    the same step group as the first because the two must come from the same
    effective-dated edition. Reading August's promotional card against July's
    reversion card is not wrong so much as unattributable (spec 6.1's cross-table
    consistency requirement, which the library does not state and this project
    needs).
    """
    pass  # tables.rate_card_20_reversion.at(risk_grade=..., limit_band=...)


@step(output="nominal_annual_rate")
def card_nominal_rate(
    promo_cell: "Row",
    reversion_cell: "Row",
    promotional_months: int,
    reversion_add_on_bps: float = overlayable(0.0, scope=["product_code", "segment_code"]),
) -> float:
    """Normalise two rates and a date into ONE absolute annual rate, for ranking only.

    The normalisation is the time-weighted rate over the mandatory 36-month
    paydown: promotional for `promotional_months`, reversion thereafter. That
    single number is what the objective and CON-INT-06 compare against the other
    three products, and it is the ONLY thing it is used for.

    `rate_components` carries the unflattened truth - {promotional_rate,
    promotional_months, reversion_rate} - into the record and onto the client's
    screen. Normalising for ranking and recording the raw representation is the
    answer to spec question 6: core.rate_card is not one capability with a
    variable shape of answer; the LOOKUP returns a row and NORMALISATION IS
    PRODUCT LOGIC. Putting normalisation in the library would mean the library
    deciding that a promotional rate is worth 8 months of a 36-month horizon,
    which is a credit policy question with a product team's name on it.
    """
    pass  # time-weighted over 36 months, plus any reversion overlay


# --- the payment that is not an instalment -----------------------------------


@step(output="minimum_payment")
def minimum_payment(
    utilised_balance: float,
    monthly_interest: float,
    monthly_fees: float,
    pct: float = param(3.0, ge=1.0, le=10.0),
    floor: float = param(50.0, ge=0.0, le=500.0),
) -> float:
    """max(R50, 3.0% of balance + interest + fees). A formula, not an instalment."""
    pass  # max(floor, balance * pct/100 + interest + fees)


@step(output="mandatory_paydown")
def mandatory_paydown(
    transferred_amount: float,
    paydown_months: int = param(36, ge=12, le=60),
) -> float:
    """At least 1/36 of the transferred amount per month, ON TOP of the minimum payment.

    Contractual, and the whole reason this product can be compared to the other
    three at all. Without it a balance transfer is an indefinite extension at a
    reverted rate and the anti-harm rule has nothing to measure - there is no
    horizon, so there is no total cost, so there is no harm and no help.
    """
    pass  # transferred_amount / paydown_months


@step(output="stressed_payment")
def stressed_payment(
    approved_limit: float,
    reversion_cell: "Row",
    paydown_months: int = param(36, ge=12, le=60),
) -> float:
    """THE REVERSION RATE ON THE FULL APPROVED LIMIT, AMORTISED OVER 36 MONTHS.

    Not the promotional minimum payment. Not the payment on the transferred
    balance. The worst case the client has actually signed up for.

    A client who can afford the R310 promotional minimum but not the R1 240
    stressed payment has been sold a cliff, and an objective minimising the
    monthly commitment will choose this product every single time if it is
    handed R310. This step is the difference between a balance transfer being an
    option and being a trap, and it is four lines of arithmetic.
    """
    pass  # amortisation of approved_limit at reversion_rate over paydown_months


@step(output="committed_monthly")
def card_committed_monthly(stressed_payment: float) -> float:
    """The contract surface. PUBLISHES THE STRESSED PAYMENT, and tests on the same one.

    contracts/product_offer.json's composition-time assertion: "an arm that tests
    affordability against a different figure from the one it publishes here is a
    defect". This arm publishes the stressed payment and AffordScenario binds
    `instalment` to this column, so the two cannot diverge without a visible
    rebind in this file.
    """
    pass  # stressed_payment


# --- no term, no balloon ------------------------------------------------------


@step(output="term_months")
def card_term_months() -> Maybe[int]:
    """na(). This product has no term, and saying so is not the same as saying zero.

    Doc 03 8.2 requires every arm to produce every declared `modifies` value with
    agreeing types. A sentinel of 0 flows into CON-INT-05 - "maximum term
    extension over the longest settled account's remaining term" - which computes
    0 - 31 = -31 and PASSES a threshold of +24. A rule intended to stop term
    extension would silently approve every balance transfer.

    na() is declared, the type is Maybe[int], and CON-INT-05 must declare it
    reads a Maybe and return na() rather than a verdict. The record then shows
    CON-INT-05 as NOT APPLICABLE on product 20, which is spec 5.7.6 exactly, and
    it shows it by construction rather than by a table of exceptions somebody
    maintains. See FRAMEWORK-DEMANDS D4.
    """
    return na()


@step(output="horizon_months")
def card_horizon() -> int:
    """36. The mandatory paydown period, which is what total cost is measured over."""
    pass  # paydown_months


@step(output="total_cost_of_credit")
def card_total_cost(
    transferred_amount: float,
    transfer_fee: float,
    promo_cell: "Row",
    reversion_cell: "Row",
    promotional_months: int,
    paydown_months: int = param(36, ge=12, le=60),
) -> float:
    """Over the mandatory paydown schedule: promotional rate, THEN reversion rate.

    Not at the promotional rate throughout, which is how this product is
    mis-sold. The transfer fee (2.5% of the transferred amount) is added to the
    utilised balance rather than paid in cash, so it accrues at both rates and
    belongs inside this calculation rather than beside it.
    """
    pass  # two-phase amortisation schedule over 36 months, summed


@step(output="product_rejection_codes")
def card_policy_gates(
    all_accounts_revolving: bool,
    transferred_amount: float,
    approved_limit: float,
    revolving_accounts_in_arrears: int,
    prior_balance_transfer_months_ago: int,
    stressed_payment_affordable: bool,
    max_transferred_pct: float = param(80.0, ge=50.0, le=95.0),
    min_limit: float = param(1000.0, ge=500.0, le=10000.0),
    max_limit: float = param(300000.0, ge=50000.0, le=500000.0),
    max_revolving_in_arrears: int = param(1, ge=0, le=3),
    min_months_since_prior_transfer: int = param(12, ge=0, le=36),
) -> list:
    """A non-revolving account in the set; transferred balance above 80% of limit;
    more than one revolving account in arrears; a prior transfer within 12 months;
    stressed payment unaffordable.
    """
    pass  # list of RJ-20-xx codes, all that apply


@step(output="conditions_precedent")
def card_conditions(settled_revolving_account_refs: list, surviving_limit: float) -> list:
    """Settled revolving accounts must be CLOSED or limit-reduced by the transferred amount.

    A condition, and it must appear in the execution package - not as a note, as
    an instruction with an owner. A balance transfer that settles three cards and
    leaves three limits open has moved R80 000 of debt and created R80 000 of
    headroom, and the client will use it. The condition is also why
    search/evaluate.py's `surviving_limit` step exists: core.obligations has to
    see the post-action state, and the post-action state depends on this
    condition being enforced rather than merely printed.
    """
    pass  # list of CP-20-xx per settled revolving account


CardBalanceTransfer = module(
    promo_cell,
    reversion_cell,
    card_nominal_rate,
    minimum_payment,
    mandatory_paydown,
    stressed_payment,
    card_committed_monthly,
    card_term_months,
    card_horizon,
    card_total_cost,
    card_policy_gates,
    card_conditions,
    name="card_20",
    params="config/products/card_20.json",
    contract="contracts/product_offer.json",
    taps=["promo_cell_id", "reversion_cell_id", "stressed_payment", "minimum_payment"],
)

# Both payment figures are tapped. The client is shown the minimum payment
# because that is what they will pay next month; the Bank decides on the stressed
# payment because that is what they will pay in month 13. Tapping only one of
# them would make the record unable to answer "what did you tell the client" and
# "what did you test" as separate questions, and an ombud asks both.
