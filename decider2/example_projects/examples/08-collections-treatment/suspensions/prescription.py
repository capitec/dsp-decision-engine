"""Suspension 114 — prescription. The one that is not really a suspension.

Spec §5.2: "Prescription is not merely a suspension. Where the debt has
prescribed, it is unenforceable, and the constraint is on what may be SAID as
much as on what may be done."

Three things make this the hardest rule in the project:

1. `prescription_date` is DERIVED, daily, per account, from the LATER of three
   events — last payment, last acknowledgement (including a recorded verbal
   acknowledgement on a call), and service of legal process — plus the
   prescription period. Any qualifying event resets it, so it cannot be stored.

2. It is a constraint on SPEECH. A blocked treatment is not enough; the script
   changes and a specific disclosure becomes mandatory. So prescription does not
   only narrow the permitted-treatment set, it narrows the permitted SCRIPT set,
   which means the script registry is part of the decision and not a downstream
   lookup. See evidence/record.py.

3. It runs the wrong way. Change scenario 10 — "a court decision requires a
   proactive written notification to clients whose debt has prescribed" — makes
   prescription a suspension that TRIGGERS a treatment rather than blocking one.
   A design where suspensions can only subtract cannot absorb it. So
   `Scope.blocks_all_except(...)` takes the notification treatment as an
   exception and the notification's own eligibility rule names suspension 114 as
   a PRECONDITION. The asymmetry is the point: a suspension may enable exactly
   one thing, named in its own declaration, on Compliance's release path.
"""

from decider2 import param, step, suspension
from decider2.types import Date, i1, i2, i4

from ..timelines import windows as W
from ..timelines.calendar import age_days, days_between
from .panel import Scope, computed

PRESCRIPTION_NOTIFICATION = 14   # change scenario 10's new treatment code


def prescription_period_days(
    product_family_code: i1,
    period_unsecured_days: i4 = param(1095, ge=365, le=10950, owner="regulatory_compliance"),
    period_mortgage_days: i4 = param(10950, ge=365, le=10950, owner="regulatory_compliance"),
) -> i4:
    pass


@step(output="prescription_date")
def prescription_date(
    last_payment_on: Date | None,
    last_acknowledgement_on: Date | None,
    last_legal_process_served_on: Date | None,
    account_opened_on: Date,
    prescription_period_days: i4,
) -> Date:
    """The later of the three interrupting events, plus the period.

    Recomputed every day for every account. Not stored, not carried, not cached:
    a qualifying event resets it and the whole point of §5.2 is that the reset
    is honoured. Cost is three max()es and an add — ~2ns/row.
    """
    pass


@step(output="days_to_prescription")
def days_to_prescription(prescription_date: Date, decision_date: Date) -> i4:
    pass  # days_between(decision_date, prescription_date); negative once prescribed


@suspension(
    code=114,
    name="prescribed",
    description="The debt has prescribed and is unenforceable. No demand may be "
                "made, no acknowledgement solicited, and the debt may not be placed. "
                "A prescribed-debt disclosure is required before any volunteered payment.",
    scope=Scope.blocks_all_except(PRESCRIPTION_NOTIFICATION),
    expiry=computed("prescription_date"),   # permanent once reached, unless interrupted before
    enables=(PRESCRIPTION_NOTIFICATION,),   # <-- the change-scenario-10 hook, declared
    restricts_scripts_to="script_family:prescribed_debt",
    owner="regulatory_compliance",
    legal_basis="Prescription Act s10/s11",
)
def prescribed(days_to_prescription: i4) -> bool:
    pass  # days_to_prescription <= 0


@step(output="pre_prescription_flag")
def pre_prescription_window(
    days_to_prescription: i4,
    flag_within_days: i4 = param(60, ge=0, le=365, owner="regulatory_compliance"),
) -> bool:
    """Not a suspension — a PRIORITY. An account worth suing that is about to
    prescribe must outrank everything for legal capacity, and that priority
    competes for the same scarce pool as everything else (spec §5.2 -> §5.9).
    Consumed by allocation/ranking.py as a mandatory-pull predicate, which is why
    it lives here and not there: the date logic belongs with the date logic.
    """
    pass
