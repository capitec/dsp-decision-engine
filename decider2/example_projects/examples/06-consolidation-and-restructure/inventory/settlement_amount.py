"""What it costs to close each settleable account, and the date that figure expires.

THE SETTLEMENT AMOUNT IS NOT THE BALANCE. The spec states this because it has
been got wrong in production by every lender that has ever built this, and the
failure mode is a shortfall discovered by the back office after disbursement,
which is an unsecured unauthorised advance.

    settlement_amount =
          outstanding capital at the assumed settlement date
        + interest accrued from the last statement to that date (per diem x days)
        + fees and charges unpaid at that date
        + early settlement charge, where permitted and applicable
        + security release, cancellation or de-registration cost, where secured
        - unearned service fees and premiums refunded on early settlement

Each of the six terms is its own step, because each is separately wrong in
production and the record must attribute a discrepancy to a NAMED COMPONENT.
When the back office settles R41 218.66 against a figure of R40 903.12 shown to
the client, "the difference is R315.54 of per-diem interest over nine days
because the payment ran three days late" is an answer; "the estimate was off" is
not.

One step per component is also the ergonomics doc 03 1.1 asks for: six functions
in one file, each readable in ten seconds, each independently testable against a
provider's own quotation. A single `compute_settlement_amount` would be the
thing a reviewer cannot check.
"""

from decider2 import module, param, step
from decider2.values import fresh_until, missing_as, na


@step(output="assumed_settlement_date")
def assumed_settlement_date(
    decision_date: "date",
    provider_payment_turnaround_days: int,
    disbursement_lag_days: int = param(2, ge=0, le=10),
) -> "date":
    """decision_date + the provider's payment turnaround + the Bank's disbursement lag.

    Ranges from 1 to 12 working days. THE AMOUNT DEPENDS ON IT, so it must be
    recorded alongside the amount. An amount without its assumed date is a number
    nobody can check, and every downstream component reads this one output.
    """
    pass  # working-day arithmetic over the provider's turnaround


@step(output="capital_at_settlement")
def capital_at_settlement(
    balance: float,
    instalment: float,
    nominal_annual_rate: float,
    statement_date: "date",
    assumed_settlement_date: "date",
) -> float:
    """Outstanding capital at the assumed settlement date, not at statement date."""
    pass  # amortise forward from the last statement to the assumed date


@step(output="per_diem_interest")
def per_diem_interest(
    capital_at_settlement: float,
    nominal_annual_rate: float,
    statement_date: "date",
    assumed_settlement_date: "date",
) -> float:
    """Interest accrued since the last statement: per diem x days.

    The reason the true amount is only knowable on the day the money arrives, and
    therefore the reason search/evaluate.py adds a settlement buffer. Recorded
    separately from the buffer so the two are never confused: the buffer is a
    policy allowance, this is arithmetic.
    """
    pass  # capital * rate/365 * days_between


@step(output="early_settlement_charge")
def early_settlement_charge(
    capital_at_settlement: float,
    nominal_annual_rate: float,
    early_settlement_rule_code: int,
    notice_given_days: int = missing_as(0),
    is_large_agreement: bool = missing_as(False),
) -> float:
    """The published mechanism: no charge below the large-agreement threshold.

    At or above it, a fixed-rate agreement may attract a charge not exceeding a
    stated number of months' interest where the required notice is not given.

    BONDS ARE THE SHARP CASE. The cancellation notice period is long enough
    (illustratively 90 days) that the charge is routinely incurred and routinely
    forgotten. Forgetting it understates the required advance by two or three
    months' interest on a bond balance, which is the largest single component in
    the whole calculation and is discovered by the back office, not here.
    """
    pass  # table lookup on rule code + notice band, capped months of interest


@step(output="security_release_cost")
def security_release_cost(
    is_secured: bool,
    security_type_code: int,
    security_releases: bool,
    security_transfers: bool,
) -> float:
    """Cost of releasing, cancelling or de-registering a security interest.

    Vehicle de-registration is small and fast. Bond cancellation is neither: 45
    to 90 working days and a conveyancing cost, which is why product 40 scenarios
    are always conditional and always require re-derivation.
    """
    pass  # tariff lookup by security type; 0.0 where unsecured


@step(output="unearned_rebate")
def unearned_rebate(
    remaining_term_months: int,
    monthly_service_fee: float = missing_as(0.0),
    credit_life_premium: float = missing_as(0.0),
    rebate_permitted: bool = missing_as(False),
) -> float:
    """Unearned service fees and credit life premium refunded on early settlement.

    CUTS THE OTHER WAY AND IS ALSO FORGOTTEN. Omitting the rebate overstates the
    required advance, which the client then pays interest on for the next five
    years. A positive number here REDUCES the settlement amount, which is why it
    is named `rebate` and subtracted at the call site rather than being returned
    negative - a sign error in this step is a five-year overcharge and the
    naming is the cheapest defence available.
    """
    pass  # unearned months x (fee + premium), where the product provides for it


@step(output="settlement_amount")
def settlement_amount(
    capital_at_settlement: float,
    per_diem_interest: float,
    unpaid_fees: float,
    early_settlement_charge: float,
    security_release_cost: float,
    unearned_rebate: float,
) -> float:
    """The six components, assembled. Every one of them is separately recorded."""
    pass  # capital + interest + fees + charge + release - rebate


@step(output="amount_basis_code")
def amount_basis_code(
    settleability_code: int,
    has_unexpired_quotation: bool,
    is_internal: bool,
) -> int:
    """QUOTED / DERIVED_INTERNAL / ESTIMATED, and the estimation tolerance where estimated.

    An estimated amount makes the whole SCENARIO conditional, so this value
    propagates up through the search: H6 orders on it, the execution package
    prints it, and offer validity is computed from the quotations behind it.
    """
    pass  # three-way on quotation presence and internality


# --- quotation expiry ---------------------------------------------------------
#
# Spec 5.10's expiry problem, and the reason `fresh_until` exists in this sketch
# rather than a staleness check in a body. FRAMEWORK-DEMANDS D12.
#
# Freshness is a property of an INPUT, so it belongs where doc 03 1 puts null
# policy: in the signature, declared, visible to a reviewer, impossible to forget.


@step(output="quotation_state")
def quotation_state(
    decision_date: "date",
    quotation_reference: str = fresh_until("quotation.expiry_date", on_stale="degrade"),
    quotation_amount: float = fresh_until("quotation.expiry_date", on_stale="degrade"),
) -> int:
    """HELD / EXPIRED / NEVER_OBTAINED, resolved at decision_date.

    `on_stale="degrade"` means an expired quotation does not fail the assessment;
    the account falls to "quotable but not quoted", its amount is estimated, and
    the outcome is marked conditional (spec 4.8). The alternative behaviours a
    declared freshness can take are "hard_stop" (internal balances, rate cards,
    the vehicle guide for product 30) and "withdraw" (the product leaves routing,
    recorded, and the assessment continues - NOT fails).
    """
    pass  # three-way on expiry vs decision_date


@step(output="offer_valid_until")
def offer_valid_until(
    earliest_quotation_expiry: "date",
    valuation_valid_until: "date",
    decision_date: "date",
    offer_validity_days: int = param(21, ge=1, le=60),
) -> "date":
    """min(earliest quotation expiry, the Bank's 21 days, any valuation validity).

    Stated PROMINENTLY on the output, because an acceptance after this date is
    not executable on the quoted figures and the client must know before they go
    away and think about it.

    On a product 40 scenario, where registration takes 6 to 10 weeks, EVERY
    quotation in the set will expire before disbursement. Re-derivation is not an
    edge case there; it is the normal path, and output/execution_package.py is
    written on that assumption rather than handling it.
    """
    pass  # min of three dates


SettlementAmounts = module(
    assumed_settlement_date,
    capital_at_settlement,
    per_diem_interest,
    early_settlement_charge,
    security_release_cost,
    unearned_rebate,
    settlement_amount,
    amount_basis_code,
    quotation_state,
    offer_valid_until,
    name="settlement_amounts",
    taps=["settlement_amount@*", "amount_basis_code"],
    contract="contracts/settlement_amount.json",
)

# taps=["settlement_amount@*"] is doc 03 7's every-version form and it earns its
# keep here: the component breakdown IS the version chain of the amount, and a
# tap per version gives the back office the attributable difference for free
# rather than as a separate reporting artefact.
