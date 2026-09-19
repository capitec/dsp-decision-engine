"""The 22 library capabilities, bound once. Spec 4.9, project 00 6.

All 22 are consumed. Fifteen are consumed inside a collection; eleven are
consumed AGAIN at every review, amendment and cascade on a subject whose data
has moved since. The per-day figures in the binding table are spec 4.9's and
they are the reason four of these bindings carry a `call_pattern=` declaration.

---------------------------------------------------------------------------
`core` is a namespace object, not a module import
---------------------------------------------------------------------------
`core.applicant_age_years` is the published NAME as an object, so roles.py can
key its `maps=` on it. Two properties fall out:

  - a typo is an AttributeError at import, not a silent rename;
  - when the library renames a published value in a major, every role that maps
    it is a build error naming the role, which is the one place a consumer
    should have to look. Doc 03 5.2's string-keyed relabel gives neither.
"""

from decider2 import consume, call_pattern, SINGLETON, COLLECTION, WHOLE
from consumed.manifest import MANIFEST, core_major

core = consume("credit-core", manifest=MANIFEST, major=core_major)

# --------------------------------------------------------------------------
# Consumed whole-assessment. Nothing interesting: these are the easy case, and
# they are the 18 of 22 that reuse costs nothing for.
# --------------------------------------------------------------------------
dates = core.dates              # 62 M resolutions/day. Every dated_table read.
bureau = core.bureau
scorecard = core.scorecard
calibration = core.calibration
risk_grade = core.risk_grade
eligibility = core.eligibility
appetite = core.appetite
rate_card = core.rate_card
instalment = core.instalment
fees = core.fees
rounding = core.rounding
reason_codes = core.reason_codes
consent = core.consent
credit_life = core.credit_life
income = core.income
deductions = core.deductions
expense_norms = core.expense_norms
obligations = core.obligations
affordability = core.affordability

# --------------------------------------------------------------------------
# The four that reuse costs something for. Each carries a `call_pattern`, which
# is a DECLARATION that this project calls the capability in a shape its owner
# did not size for, plus the equivalence obligation that makes the declaration
# safe rather than merely honest (spec 5.17.5, 8.2 "partial equivalence").
# --------------------------------------------------------------------------

adverse_events = core.adverse_events.with_(
    call_pattern(
        SINGLETON,
        volume="9 400/day as singletons, plus 570 000 in the monthly batch",
        designed_for="38 per application, 900 applications a day",
        # Spec 5.17.6 fork pressure #1. The cheapest thing the team could do is
        # write a stripped-down classifier for the daily pass. What stops it:
        # the component publishes a single-element interface that IS the same
        # implementation, and `equivalent_to=WHOLE` generates the test.
        equivalent_to=WHOLE,
        proof="weekly full reconciliation pass, spec 5.6.3 requirement 2",
    ),
    # Role parameterises the thresholds; the capability stays ignorant of
    # criticality classes (05 13-Q3, spec 5.17.1 property 2).
    parameterised_by="criticality_class",
)

exposure = core.exposure.with_(
    call_pattern(
        COLLECTION,
        volume="26 000/day plus every cascade; groups of mean 2.3, p99 41",
        designed_for="a client-level aggregate at one point in time",
    ),
    # Spec 5.17.2: time and contingency are BOTH absent from the published
    # capability. Declared as gap `exposure_as_at_date`, resolution EXTEND,
    # because a dated aggregate over a composition-dated set is general and
    # projects 07 and 10 will want it. Fork pressure #6 is here.
    gaps=["exposure_as_at_date", "contingent_conversion_factors"],
)

adjustments = core.adjustments.with_(
    call_pattern(
        WHOLE,
        volume="380 000 overlay evaluations/day over the 11-position stack",
        designed_for="the stack in force at one date",
    ),
    # Doc 00 6.22 produces the stack in force. The cause decomposition needs the
    # DIFFERENCE between two stacks a year apart, decomposed by overlay. Two
    # dates, not one. Gap `overlay_stack_delta`, resolution EXTEND.
    gaps=["overlay_stack_delta"],
)

# `core.reason_codes` is consumed unchanged as a capability but this project
# contributes 260 codes (ranges 5600-5999) and asks for a NEW ATTRIBUTE on the
# registry: what may be disclosed to business B about business A (spec 5.12.4,
# 9.3). Project 05's registry answers what may be said to a business about an
# individual; a third business is a third party of a different kind.
reason_codes = reason_codes.contributes(
    ranges=[(5600, 5999)],
    requires_attribute="communicable_to_connected_business",
    owner="compliance",
    gap="third_party_disclosure_class",
)
