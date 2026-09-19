"""Rule-local derived quantities — registered steps, referenced from rule
documents by id.

Spec §5.10: 3% of the live set computes "a ratio, a difference of two features".
That is ~16 rules of 521. Doc 08 §3.2 settles how they are expressed: a derived
value is a **step in code**, registered and referenced by id from a rule — never
an expression string in a rule document.

I am keeping that decision, and it costs exactly what doc 08 says it costs: an
analyst who wants a new ratio needs an engineer and a deploy, where the other
505 rules need neither. Two things make that survivable, and neither is in
doc 08:

 1. The catalogue is **pre-stocked**. Ratios, differences and log-ratios over
    the velocity cross-product are generated from the same declaration that
    generates the 168 aggregates, so the analyst's ratio usually already exists.
    Only a genuinely novel shape needs code.
 2. The rule validator's error **names the closest registered feature** and, if
    none is close, prints the exact `@feature` stub to send to engineering. The
    analyst's dead end becomes a pull request.

Without those two, doc 08 §3.2 is the rule that gets routed around — an analyst
under attack at 19:40 on a Friday will inline a literal into an adjacent rule
rather than wait for a deploy, and that is the 546-`pl.lit()` failure again.
"""

from __future__ import annotations

from decider2 import Observed, feature
from decider2.types import Money


@feature(id="feat:amount_to_avg_30d_ratio", unit="ratio")
def amount_to_avg_30d_ratio(
    amount_zar_cents: Money,
    client_amount_sum_30d: Observed[float],
    client_count_30d: Observed[int],
) -> Observed[float]:
    """This amount as a multiple of the client's 30-day average payment.

    ABSENT propagates: if either aggregate is absent the ratio is absent, and
    every rule referencing it follows its own declared on_absent behaviour. A
    derived feature never invents a value its inputs did not have.
    """
    pass


@feature(id="feat:beneficiary_concentration_24h", unit="ratio")
def beneficiary_concentration_24h(
    beneficiary_amount_sum_24h: Observed[float],
    client_amount_sum_24h: Observed[float],
) -> Observed[float]:
    """Share of the client's 24-hour outflow going to this one beneficiary."""
    pass


@feature(id="feat:device_age_vs_beneficiary_age", unit="hours")
def device_age_vs_beneficiary_age(
    hours_since_device_first_seen: Observed[float],
    beneficiary_age_hours: Observed[float],
) -> Observed[float]:
    """Signed hours between the device appearing and the beneficiary being added.

    Near zero means both were created in the same session, which is the takeover
    -then-payout signature. The AT family's highest-precision single feature.
    """
    pass


@feature(id="feat:auth_decline_ratio_10min", unit="ratio")
def auth_decline_ratio_10min(
    card_count_10min: Observed[int],
    card_declined_count_10min: Observed[int],
) -> Observed[float]:
    """Card-testing signature. Absent denominator is absent, not 1.0."""
    pass
