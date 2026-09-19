"""Applicability predicates for the cap register.

Doc 08 §3.2 settles it: a derived value is a registered STEP referenced from a
rule by id, never an expression string in config.  This file is the vocabulary
the 52 rules are allowed to speak, and growing it is the one thing in the
register that costs a release.

That price is the design working as intended -- but it has a sharp edge worth
naming.  Four owners edit this register on a quarterly cadence.  If every new
rule needs a new predicate, "configuration change, not a release" is a fiction.
The mitigation is that the predicates are DELIBERATELY GENERIC over their
thresholds: `months_employed_below(threshold)` serves CAP-0175's three tiers
and any future tenure rule, because the threshold is a param reference in the
rule row (`{"param": "..."}`), not a constant in the predicate.  Sixteen
predicates cover 52 rules today.  Whether that holds at 80 rules is the real
test of this design and it is unresolved -- FRAMEWORK-DEMANDS #5.
"""

from __future__ import annotations

from decider2 import predicate, predicate_registry
from decider2.money import Money


@predicate(id="grade_at_or_worse_than")
def grade_at_or_worse_than(risk_grade: int, threshold: int) -> bool:
    """The assigned grade is `threshold` or worse."""
    pass


@predicate(id="months_employed_below")
def months_employed_below(months_employed: "Maybe[float]", threshold: float) -> bool:
    """Employer tenure below a threshold. A null tenure does NOT satisfy this --
    9% of self-employed applicants have no tenure and must not be caught by a
    rule written for short-tenure salaried staff."""
    pass


@predicate(id="arrears_in_window")
def arrears_in_window(
    worst_arrears_months: int, months_since_worst_arrears: "Maybe[float]",
    min_arrears_months: int, window_months: float,
) -> bool:
    """Any account `min_arrears_months`+ in arrears within the last `window_months`."""
    pass


@predicate(id="enquiry_velocity_at_or_above")
def enquiry_velocity_at_or_above(enquiry_velocity_60d: int, threshold: int) -> bool:
    """Credit enquiries in the trailing 60 days at or above a threshold."""
    pass


@predicate(id="employer_on_watchlist")
def employer_on_watchlist(employer_id: "Maybe[int]", tables) -> bool:
    """Employer or its sector appears on the exposure watchlist (~2 400 entries)."""
    pass  # a set-membership lookup, monthly cadence, Credit Systems


@predicate(id="group_exposure_headroom")
def group_exposure_headroom(
    group_exposure_limit: Money, internal_exposure_total: Money
) -> Money:
    """Headroom = limit less existing internal exposure. A VALUE predicate: the rule
    reduces to this rather than testing it, which is why the register's effect
    vocabulary takes a value expression as well as a constant."""
    pass


@predicate(id="channel_in")
def channel_in(channel_code: int, values: "set[int]") -> bool:
    """Channel membership. Serves CAP-0025, CAP-0330 and any future channel rule."""
    pass


@predicate(id="campaign_authorised")
def campaign_authorised(campaign_id: "Maybe[int]", tables) -> bool:
    """A valid `campaign_id` naming a Credit-Committee-authorised pre-approved campaign."""
    pass


# ... eight more: segment_in, new_to_bank, age_at_maturity_exceeds,
# purpose_in, joint_application, insolvency_seasoning_below,
# internal_tenure_below, product_maximum.

REGISTRY = predicate_registry(
    grade_at_or_worse_than,
    months_employed_below,
    arrears_in_window,
    enquiry_velocity_at_or_above,
    employer_on_watchlist,
    group_exposure_headroom,
    channel_in,
    campaign_authorised,
    # ... 8 more
)
