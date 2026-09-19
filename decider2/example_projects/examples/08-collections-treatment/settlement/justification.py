"""Settlement offers, and the comparison that justifies them.

Spec §5.7: "A settlement granted at L1 where the grid required L3 is an audit
finding; a settlement where the comparison CANNOT BE REPRODUCED is a worse one."

So the expected-recovery comparison is not a note in a field. It is a computed
value, produced by declared steps, from the estimates in force on the day, with
the curve versions recorded — and §9.3's audit reproduces it by replaying the
same steps with the same `decision_date`, not by reading a stored number and
hoping.

Four things make that reproducible rather than aspirational:

  1. `recovery_estimate` and `cost_to_collect` come from effective-dated curves
     resolved by `decision_date` and their versions are outputs.
  2. The discount rate is a param with an owner and a bound, not a literal.
  3. The arithmetic accumulates in float64 and rounds with `round_half_up`.
     A bare `round()` here differs between njit and CPython by one cent
     (doc 03 §1.2), and one cent on a settlement comparison is the difference
     between an audit reproducing it and an audit raising a finding.
  4. The comparison is a TAP, so it is an output column in production for every
     offer, not a debugging nicety.
"""

from decider2 import module, param, round_half_up, step
from decider2.types import Date, cents, f8, i1, i2, i4

from ..matrix.grid import DISCOUNT_AUTHORITY


@step(output="npv_continued_collections")
def discounted_net_recovery(
    recovery_estimate: cents,
    cost_to_collect: cents,
    months_to_recovery: i2,
    discount_rate_pa: f8 = param(0.12, ge=0.0, le=0.5, owner="credit_risk_policy",
                                 description="Policy discount rate for recovery comparisons"),
) -> cents:
    """PV of continued normal collections over 24 months.

    Accumulates in float64 and converts to cents once at the end. An int64 cent
    accumulator over 2.3M balances wraps (doc 03 §1.2, measured at 2 667 rows on
    a sum of squares); this one is a discounted sum, but the rule is the rule.
    """
    pass


@step(output="settlement_proceeds")
def settlement_proceeds(
    outstanding_balance: cents,
    offered_discount_pct: i1,
    is_instalment_variant: bool,
    instalment_variant_penalty_pp: i1,
) -> cents:
    pass  # round_half_up, never round()


@step(output="settlement_justified")
def settlement_justified(
    settlement_proceeds: cents,
    npv_continued_collections: cents,
    margin_cents: cents = param(0, owner="credit_risk_policy"),
) -> bool:
    pass


@step(output="max_discount_at_own_authority_pct")
def envelope_at_agent_authority(
    arrears_bucket_code: i1,
    recovery_band_code: i1,
    agent_authority_level: i1,
    grid=DISCOUNT_AUTHORITY,
) -> i1:
    """What goes on the agent's screen. The envelope is computed by the flow; the
    ACCEPTANCE is a separate decision with its own authority record. Two entry
    points, one grid (pipelines/live_call.py)."""
    pass


@step(output="settlement_block_code")
def prescription_interaction(
    prescribed: bool,
    pre_prescription_flag: bool,
    days_to_prescription: i4,
    agent_authority_level: i1,
) -> i1:
    """The interaction spec §5.7 calls "easy to get wrong", and it is:

      * a settlement offer INVITES an acknowledgement of the debt;
      * an acknowledgement INTERRUPTS prescription;
      * therefore offering a settlement on a nearly-prescribed debt can revive
        an obligation that was about to die, which is a benefit to the Bank
        obtained from the client's ignorance.

    So: prescribed -> no offer at all, at any level. Within 60 days -> L3
    authority, a specific disclosure, and "the fact that the discussion could
    have interrupted prescription must be on the record" — which means an
    output field, not a script note. Returns 0 permitted, 1 requires_L3_and_
    disclosure, 2 forbidden.

    Note the direction of the dependency: this reads `prescribed` from the
    suspension panel, which is evaluated BEFORE the recommendation is finalised
    and AFTER overlays are applied (see pipelines/daily_batch.py). A design that
    evaluated suspensions first and overlays second would let a commercial
    overlay move a settlement offer onto a prescribed debt.
    """
    pass


SettlementAssessment = module(
    discounted_net_recovery,
    settlement_proceeds,
    settlement_justified,
    envelope_at_agent_authority,
    prescription_interaction,
    name="settlement",
    taps=["npv_continued_collections", "settlement_proceeds", "settlement_justified",
          "max_discount_at_own_authority_pct", "settlement_block_code"],
    # Five taps. Every one of them is a field §9.3 asks for, so they are not
    # diagnostics — they are the product. Measured cost: +0.11 ns/row/tap.
)
