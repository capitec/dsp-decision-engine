"""EP-8 -- the group cascade. Spec 5.12, H5, 13-Q9, 13-Q10.

    An entity's adverse event propagates to every facility of every business
    that entity touches -- bounded, and explainable to the client of business B
    whose facility was cut because of something at business A.

~1 900 cascades a day touching ~14 000 facilities, p95 15 minutes from the
triggering event.

---------------------------------------------------------------------------
13-Q9: is boundedness a property of the construct or a discipline applied to it?
---------------------------------------------------------------------------
A property of the construct. All four bounds are ARGUMENTS to `Cascade`, which
means a cascade with no fan-out cap does not construct, and `hops=3` is a code
change with a review rather than a loop that happens to run longer. Spec 10
acceptance 15 -- "enforced, not monitored" -- is then a statement about the type
rather than about an alerting rule.
"""

from decider2 import Cascade, queue_to, module
from decider2.frame import Join
from consumed.p05_origination import EntityAssessment, PeopleBlend, CombinedGrade, GroupExposure
from modules.appetite.facility_types import Appetite
from modules.watchlist.signals import Watchlist
from time.bitemporal import ACTUAL

GroupCascade = Cascade(
    "group_cascade",
    triggers=[
        ("adverse_event_material_or_worse", 1_100),
        ("entity_grade_moved_2_notches", 640),
        ("ownership_change", 370),          # transfer, allotment, BO restatement
        ("appointment_or_resignation", 760),
        ("facility_change_in_group", 320),
        ("covenant_breach_material_or_severe", 190),
        ("watchlist_escalation_to_w3", 140),
        ("insolvency_filing", 40),
        ("screening_status_change", 90),
        # A tenth, added by this project and not in spec 5.12.1's nine:
        ("collateral_revaluation", 0),      # modules/security/allocation.py
    ],

    # --- the four bounds, spec 5.12.2 -------------------------------------
    hops=2,                # entity -> business -> that business's group.
                           # NOT from the group's members to THEIR groups:
                           # transitive closure over a connected-client graph
                           # reaches 40% of the book from most starting points.
    materiality="the trigger must be capable of changing something",
                           # ~71% of candidate triggers stop here and propagate
                           # as a FLAG on the affected records, not a re-assessment
    fanout_cap=250,
    on_cap=queue_to("portfolio_management", window="next scheduled", named_owner=True),
                           # the 340 entities touching >100 facilities are exactly
                           # the cases where inline propagation would be both
                           # slowest and most consequential
    dedupe_on=["cascade_id", "subject_key"],
                           # re-entrancy. Cross-holdings produce loops; 05 5.1
                           # already truncates cycles INSIDE one structure, and
                           # this is the same problem BETWEEN structures.

    # --- what is re-assessed and what is only flagged, spec 5.12.3 --------
    # "getting it wrong in either direction is a failure: too much
    #  re-assessment and the book churns; too little and the Bank's exposure
    #  numbers are stale."
    recompute={
        "triggering_entity": EntityAssessment.partial(unit="entity"),
        "businesses_where_critical_or_significant": PeopleBlend.partial() | CombinedGrade,
        "group_exposure": GroupExposure,
        "appetite_headroom": Appetite,
        "watchlist_grade": Watchlist,
    },
    flag_only={
        "businesses_where_peripheral": "flag for next review",
        "contracted_facility_price": "flag for next review or an amendment",
        "drawn_committed_limit": "flag; an amendment decision with its own "
                                 "authority and client notice is required",
    },
    never={
        "contracted_facility_price": "changed automatically",
        "drawn_committed_limit": "reduced automatically",
    },
    immediate={
        "undrawn_amounts": "blocked where headroom is exceeded",
        "new_lending": "blocked where headroom is exceeded",
    },

    # --- the record, spec 5.12.5 ------------------------------------------
    # "Not a log line. A cascade has an identity, a trigger, a scope, a set of
    #  consequences and a bound that it either respected or hit."
    record=[
        "cascade_id", "trigger", "trigger_classification", "classifying_rule",
        "triggering_entity", "business_a", "propagation_path",
        "subjects_touched", "recomputed_per_subject", "before_and_after",
        "bound_applied", "decisions_produced",
    ],
    retention="7 years",
)

# The ACTUAL view, at the event date. A cascade asks "what is true now", not
# "what did we know then" -- it is a forward decision, not a replay. Its own
# replay later will use the knowledge date recorded on it, which is the
# determination date of the cascade rather than of the trigger.
GroupCascade = GroupCascade.as_true_at("event_date").reading(ACTUAL)

# --------------------------------------------------------------------------
# Two explanations, produced together, NEVER MERGED. Spec 5.12.4, 9.3.
#
# Business B's headroom drops on 14 June because a judgment was registered
# against a director of business A on 9 June, and A and B share that director.
#
# The internal attribution is complete. The communicable explanation may not
# disclose business A's affairs or the director's personal credit record, and
# says something in the shape of: "the Bank's aggregate exposure to a group of
# connected businesses of which yours is one has reached its limit" -- plus the
# route by which the individual concerned may obtain their own reasons directly
# (05 9.2, unchanged).
#
# `dual_attribution` is not a formatting choice. It makes the two artefacts two
# OUTPUTS with two declared disclosure classes, so a renderer cannot reach the
# internal one, and the adversarial test in spec 10 acceptance 16 has something
# to attack. A single record with a "communicable" flag per field is the shape
# that leaks, because a new field defaults to something.
# --------------------------------------------------------------------------
GroupCascade = GroupCascade.dual_attribution(
    internal="complete; Audit and the regulator; 7 years",
    communicable="reason registry client-facing wording, Compliance-owned",
    disclosure_class="communicable_to_connected_business",   # the NEW registry
                                                             # attribute, gap
                                                             # third_party_disclosure_class
    default="withheld",     # a new field is NOT communicable until classified
)
