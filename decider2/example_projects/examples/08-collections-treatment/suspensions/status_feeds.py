"""Suspensions sourced from the status and legal feed: 101-111, 113.

Every one of these carries its SOURCE RECORD through to the evidence — the debt
review case number, the complaint reference, the ombud case id. Spec §5.2
requires the record to name "the source record and its identifier", which means
the predicate cannot be a bare boolean over a flag column; it has to travel with
the row that caused it. That is what `carry=` on the status windows
(timelines/windows.py) exists for.

The registry short-delivers (spec §4.2). That is not an error state, it is a
KNOWLEDGE state, and it is why every one of these declares `feed=` on its
event-driven expiry: §9.2's answer — "the status arrived after the run, and here
is the feed watermark that proves it" — is assembled from the declaration, not
written by hand at complaint time.
"""

from decider2 import param, suspension
from decider2.types import Date, Timestamp, i1, i2

from ..timelines import windows as W
from .panel import Scope, computed, event

TREATMENTS_ARRANGEMENT_SERVICING = (0, 12)
STATUTORY_NOTICES_ONLY = (10,)
EXECUTOR_CORRESPONDENCE = (10,)


@suspension(
    code=101,
    name="debt_review_application",
    description="Debt review application received. All collections contact stops "
                "except statutory notices.",
    scope=Scope.blocks_all_except(*STATUTORY_NOTICES_ONLY),
    expiry=event(awaits="debt_review.stage_change", reviewed="daily",
                 feed="status_events"),
    owner="regulatory_compliance",
    legal_basis="NCA s86",
)
def debt_review_application(debt_review_stage: i2 | None) -> bool:
    pass  # stage 1 and uncleared


@suspension(
    code=102,
    name="debt_review_proposal",
    description="Debt review proposal issued. All collections except arrangement servicing.",
    scope=Scope.blocks_all_except(*TREATMENTS_ARRANGEMENT_SERVICING),
    expiry=event(awaits="debt_review.stage_change", reviewed="daily", feed="status_events"),
    owner="regulatory_compliance",
)
def debt_review_proposal(debt_review_stage: i2 | None) -> bool:
    pass  # stage 2 and uncleared


@suspension(
    code=103,
    name="debt_review_order",
    description="Debt review court order granted. Rearrangement servicing only.",
    scope=Scope.blocks_all_except(*TREATMENTS_ARRANGEMENT_SERVICING),
    expiry=event(awaits="debt_review.order_set_aside_or_clearance", reviewed="daily",
                 feed="status_events"),
    owner="regulatory_compliance",
)
def debt_review_order(debt_review_stage: i2 | None) -> bool:
    pass  # stage 3 and uncleared


@suspension(
    code=104,
    name="debt_review_rearrangement_default",
    description="Rearrangement in default. Releases 103 after the prescribed process.",
    scope=Scope.blocks_treatments(11),          # legal still blocked until the process completes
    expiry=computed("add_bdays(rearrangement_default_on, params.release_business_days, bday)"),
    owner="regulatory_compliance",
    releases=(103,),    # DECLARED interaction. The panel checks that a rule which
                        # releases another is evaluated in the same panel and that
                        # the release is recorded as its own evidence row.
)
def debt_review_rearrangement_default(
    rearrangement_default_on: Date | None,
    release_business_days: i2 = param(20, ge=1, le=60, owner="regulatory_compliance"),
) -> bool:
    pass


@suspension(
    code=105, name="administration",
    description="Administration order granted. All normal collections stop.",
    scope=Scope.blocks_all_except(*STATUTORY_NOTICES_ONLY),
    expiry=event(awaits="administration.discharge", reviewed="weekly", feed="status_events"),
    owner="regulatory_compliance",
)
def administration(administration_order_on: Date | None, administration_cleared_on: Date | None) -> bool:
    pass


@suspension(
    code=106, name="insolvency",
    description="Insolvency or sequestration. All activity stops until rehabilitation.",
    scope=Scope.blocks_all(),
    expiry=event(awaits="insolvency.rehabilitation", reviewed="weekly", feed="status_events"),
    owner="regulatory_compliance",
)
def insolvency(sequestration_on: Date | None, rehabilitation_on: Date | None) -> bool:
    pass


@suspension(
    code=107, name="deceased_estate",
    description="Client deceased, estate under administration. Executor correspondence only.",
    scope=Scope.blocks_all_except(*EXECUTOR_CORRESPONDENCE),
    expiry=event(awaits="estate.finalised", reviewed="weekly", feed="status_events"),
    owner="regulatory_compliance",
)
def deceased_estate(deceased_notified_on: Date | None, estate_finalised_on: Date | None) -> bool:
    pass


@suspension(
    code=108, name="internal_complaint",
    description="Active internal complaint. Contact and legal suspended.",
    scope=Scope.blocks_channels(1, 2, 3, 4, 5, 6) | Scope.blocks_treatments(11),
    expiry=computed("add_bdays(complaint_logged_on, params.complaint_business_days, bday)"),
    owner="regulatory_compliance",
)
def internal_complaint(
    complaint_logged_on: Date | None,
    complaint_closed_on: Date | None,
    complaint_extended_to: Date | None,
    complaint_business_days: i2 = param(15, ge=5, le=60, owner="regulatory_compliance"),
) -> bool:
    pass  # extendable: expiry is max(computed, complaint_extended_to)


@suspension(
    code=109, name="ombud_referral",
    description="Ombud or regulator referral open. Contact and legal suspended.",
    scope=Scope.blocks_channels(1, 2, 3, 4, 5, 6) | Scope.blocks_treatments(11),
    expiry=event(awaits="ombud.case_closed", reviewed="daily", feed="status_events"),
    owner="regulatory_compliance",
)
def ombud_referral(ombud_case_opened_on: Date | None, ombud_case_closed_on: Date | None) -> bool:
    pass


@suspension(
    code=110, name="dispute_raised",
    description="Dispute raised on the debt. Legal and agency handover blocked; "
                "contact permitted in writing.",
    scope=Scope.blocks_treatments(9, 11) | Scope.blocks_channels(5, 6),
    expiry=event(awaits="dispute.resolved", reviewed="daily", feed="status_events"),
    owner="regulatory_compliance",
)
def dispute_raised(open_dispute_ref: str | None) -> bool:
    pass


@suspension(
    code=111, name="hardship_performing",
    description="Hardship arrangement in good standing. All collections stop while performing.",
    scope=Scope.blocks_all_except(*TREATMENTS_ARRANGEMENT_SERVICING),
    expiry=event(awaits="arrangement.first_missed_instalment", reviewed="daily",
                 feed="arrangement_instalments"),
    owner="credit_risk_policy",
)
def hardship_performing(
    hardship_arrangement_id: i2 | None,
    missed_instalments_current_arrangement: i2,
) -> bool:
    pass


@suspension(
    code=113, name="litigation_in_progress",
    description="Litigation in progress. All non-legal treatments blocked.",
    scope=Scope.blocks_all_except(10, 11),
    expiry=event(awaits="litigation.judgment_or_withdrawal", reviewed="weekly",
                 feed="status_events"),
    owner="recoveries_and_legal",
)
def litigation_in_progress(litigation_opened_on: Date | None, litigation_closed_on: Date | None) -> bool:
    pass
