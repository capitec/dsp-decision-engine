"""Stage 5.11 -- disclosure outputs.

Two requirements that look like formatting and are not.

1.  **Five figures that must sum to the total.**  `total_cost_of_credit` breaks
    into capital, interest, initiation fee, service fees and premiums.  A
    rounding convention applied independently to five components does not sum
    to the independently-rounded total -- this is the classic cent-level
    reconciliation break, and it is why `core.rounding` is a published
    capability rather than a convention.  `sums_to=` declares the constraint
    and the framework asserts it, so the break is a build-time or runtime
    failure rather than a monthly finance query.

2.  **On decline, the reason set is assembled from every stage that
    contributed.**  Gate failures, fraud decline, policy rule declines,
    affordability failure, and the scorecard's top negative contributions where
    the decline was score-driven.  Ranked by the registry's severity order,
    capped at four communicated, ALL recorded, in the client's language of
    record.

    The assembly is a fold over sources that each already produced a reason
    vector -- `gate_verdicts`, `cap_rule_verdicts`, `score_reason_codes`,
    `suppression_reason_codes`.  It is not a re-derivation.  That matters
    because §9.1 requires the reasons to be retrievable for an application
    declined months ago, and a reason assembled at display time from a decision
    record that did not store its inputs cannot be.
"""

from __future__ import annotations

from decider2 import module, param, step
from decider2.money import Money


@step(output="cost_breakdown")
def cost_breakdown(
    offered_amount: Money, total_cost_of_credit: Money, nominal_annual_rate: int,
    initiation_fee_incl_tax: Money, monthly_service_fee: Money,
    credit_life_premium: Money, term_months: int,
) -> "Breakdown[5]":
    """Capital, interest, initiation fee, service fees, premiums."""
    pass


Quotation = module(
    cost_breakdown,
    name="quotation",
    # The constraint, declared.  Checked in every execution mode.
    invariants=["cost_breakdown.sums_to(total_cost_of_credit)"],
    writes=["offered_amount", "initiation_fee", "initiation_fee_tax", "amount_financed",
            "nominal_annual_rate", "monthly_service_fee", "credit_life_premium",
            "credit_life_substitution_right_disclosed", "instalment", "instalment_count",
            "first_payment_amount", "first_payment_date", "final_payment_amount",
            "final_payment_date", "total_cost_of_credit", "cost_breakdown",
            "effective_annual_rate", "quotation_valid_until"],
)


@step(output="decline_reason_codes")
def assemble_reasons(
    gate_verdicts: "int8[16]",
    cap_rule_verdicts: "int8[64]",
    score_reason_codes: "int16[4]",
    suppression_reason_codes: "int16[8]",
    fraud_reason_codes: "int16[8]",
    affordability_verdict_code: int,
    tables,
    reasons_communicated: int = param(4, ge=1, le=8, owner="compliance"),
) -> "int16[32]":
    """Every reason that applied, ranked by the registry's severity order.

    `primary_reason_code` is the first.  Up to `reasons_communicated` are
    communicated; ALL are recorded.  Where the decline was score-driven the
    communicated reasons are the characteristic contributions in the registry's
    client-facing wording, resolved to the client's language of record.
    """
    pass


Reasons = module(assemble_reasons, name="reasons",
                 taps=["primary_reason_code", "decline_reason_codes"])
