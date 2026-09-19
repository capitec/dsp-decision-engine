"""Suspension 112 — prescribed notice period unexpired.

The only suspension in the set whose expiry is pure arithmetic, and therefore
the one that shows what `expiry=computed(...)` buys. Notice delivered
14 August + 10 business days = 28 August, computable today, computable in 2033,
computable for every account in the book in one vectorised subtract.

Acceptance criterion 12 attaches here and it is a STRUCTURAL requirement, not a
rule: "legal handover cannot occur without the notice evidence being present,
and the attempt to hand over without it is itself recorded." So the notice gate
is not a predicate returning False. It is a `precondition` on treatment 11
declared beside the treatment itself, and the framework emits the attempted-
handover record when the precondition fails. A rule you can forget to write is
the wrong mechanism for a thing the regulator tests exhaustively (§9.4).
"""

from decider2 import param, precondition, suspension
from decider2.types import Date, Timestamp, i1, i2

from ..timelines import windows as W
from ..timelines.calendar import add_bdays, bdays_between
from .panel import Scope, computed

# Notice period requirements: 8 products x 4 notice types x 6 attributes.
# Compliance-owned, effective-dated, changed on regulatory change. Change
# scenario 4 — "the regulator shortens the prescribed notice period for one
# product family" — is a new effective-dated version of this table and nothing
# else. It is NOT a param, because it is keyed by product family and notice
# type; it is a Table (matrix/grid.py's `Grid`, one dimension smaller).
from ..matrix.grid import NOTICE_PERIODS


@suspension(
    code=112,
    name="notice_period_unexpired",
    description="A prescribed pre-legal notice has been delivered but the "
                "statutory period has not yet elapsed. Legal handover only is blocked.",
    scope=Scope.blocks_treatments(11),
    expiry=computed("notice_period_expires_on"),
    owner="regulatory_compliance",
    legal_basis="NCA s129/s130",
)
def notice_period_unexpired(
    notice_delivered_on: Date | None,
    notice_period_expires_on: Date | None,
    decision_date: Date,
) -> bool:
    pass  # delivered, and decision_date < expiry


def notice_period_expires_on(
    notice_delivered_on: Date | None,
    product_family_code: i1,
    notice_type_code: i1,
    bday,
) -> Date | None:
    """delivery date + the product family's prescribed business days.

    One line, and it is the entire answer to the "computable expiry" half of
    spec §13.8. The other half — event-driven expiry — cannot be an expression
    and is handled by `expiry=event(...)` in status_feeds.py. There are exactly
    two kinds and the type says so.
    """
    pass  # add_bdays(notice_delivered_on, NOTICE_PERIODS[family, type].business_days, bday)


@precondition(
    treatment_code=11,
    name="legal_handover_requires_served_notice",
    description="No legal handover without delivery proof and an elapsed notice period.",
    on_failure="record_attempt",     # <-- emits an attempted-handover evidence row
    severity="blocking",
)
def legal_handover_notice_evidence(
    notice_delivered_on: Date | None,
    notice_delivery_proof_ref: str | None,
    notice_period_expires_on: Date | None,
    decision_date: Date,
) -> bool:
    """§9.4: evidence for EVERY account handed to legal in a period, not a sample."""
    pass


@precondition(
    treatment_code=9,
    name="agency_handover_requires_no_dispute",
    description="A disputed debt may not be placed with a third party.",
    on_failure="record_attempt",
    severity="blocking",
)
def agency_handover_dispute_check(open_dispute_ref: str | None, prescribed: bool) -> bool:
    pass
