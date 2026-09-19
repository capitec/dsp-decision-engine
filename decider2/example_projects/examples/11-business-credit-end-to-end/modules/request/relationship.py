"""O1 -- request and relationship resolution. The phase project 05 does not have.

Spec 5.3 O1. Project 05's applicant either exists or does not. Here the client
may have eleven facilities, three of them on watchlist, one forborne and one in
a cross-default event nobody has called.

Two jobs, and the second is the one everything else depends on.
"""

from decider2 import module, step, reserved_input
from time.bitemporal import EntityFacts

# --------------------------------------------------------------------------
# JOB 1: the four dates, set here, separate, and none of them is the run date.
#
# Spec 5.3 O1: for EP-3 `decision_date` is the REVIEW DATE, which may precede the
# run date by up to 30 days. For EP-4 it is the COVENANT'S TEST DATE, which may
# precede the run by 90 days because the certificate arrived late.
#
#   "A phase that reads the run date is wrong in both cases and WILL REPLAY
#    SUCCESSFULLY WITH THE WRONG ANSWER." (09 5.15 item 4)
#
# `reserved_input` is doc 07's `decision_date` treatment generalised to four.
# Each is frame-only: no step may produce one, so nothing can overwrite one
# mid-flow, and every dated artefact resolves against the right one. Omitting
# any from the input schema is a build failure listing every artefact that
# needed it.
#
# There is no `.today()`, no `.now()` and no `datetime` import anywhere under
# modules/ or pipelines/ -- a lint, in the shape of doc 07 6's existing rules.
# --------------------------------------------------------------------------
decision_date = reserved_input("decision_date", doc="Whose rules govern. 00 7.3.")
knowledge_date = reserved_input("knowledge_date", doc="As-at of the Bank's view. Spec 5.13.")
test_date = reserved_input("test_date", doc="As-at of a covenant measurement. Spec 5.5.2.")
determination_date = reserved_input("determination_date", doc="When the flow ran. What 09 replays against.")

# --------------------------------------------------------------------------
# JOB 2: load existing state and make it available to every later phase.
#
# Four of these are AUTHORITY MODIFIERS (spec 5.14) and one -- the forbearance
# flag -- changes what L5 is permitted to do. So they are not context; they are
# inputs to the decision, and they are loaded first or the authority is wrong.
# --------------------------------------------------------------------------


def subject_kind(application_id, facility_id, client_id, group_id) -> int:
    """Four subject kinds and the flow must be able to say which it has.

    Application / Facility / Client / Group. `group_id` is NOT STABLE and that
    is a stated requirement rather than a defect (spec 4.1): a group is a
    derived set, and pretending it has a stable identity is how a 2031 group
    exposure report silently includes a business that left the group in 2028.

    So a group subject is identified by (member set, composition date) -- see
    roles.py GROUP's `identity=` -- and every group-level output names that date.
    """
    pass  # exactly one of the four must be non-null


def facility_decision_history(facility_id: int, knowledge_date) -> list:
    """1..~40 decisions of record, immutable. Mean 11 over five years, p99 58.

    Spec 13-Q2 asks where the history lives: a parameter, an input, a store the
    flow reads, or a first-class subject the flow operates on -- "and what
    happens to replay if it is a store?"

    This project's answer: an INPUT, always, never a store read. The predecessor
    decision of record is a row on the input frame, resolved by the caller,
    captured in the evidence record like any other input (09 5.15 item 8:
    "mutable state snapshotted or addressable"). A flow that reaches into a
    decision store during execution cannot replay without that store's history,
    and the store does not retain history -- which 09 5.15 item 8 names as a
    replay defect in projects 07 and 08 today.

    The cost is real and is stated in FRAMEWORK-DEMANDS D5: the caller must
    resolve the predecessor before the flow runs, which means the 8-hour review
    batch does a 15 000-row join before it decides anything. That is a frame-tier
    join, it is cheap, and it is the correct place for it.
    """
    pass  # the predecessor row, as an input column


def existing_state(facility_id: int, knowledge_date) -> dict:
    """watchlist_grade, forbearance_flag + probation clock, staging_code,
    cross-default state, live waivers, review status.

    Spec 13-Q7: "what is the unit of state?" Four flags per facility, with
    probation clocks, permitted transitions and interactions THAT DO NOT IMPLY
    EACH OTHER. A facility can be W2, not forborne, performing and Stage 2. It
    can be W0, forborne under probation, performing and Stage 2.

    This project's answer: the flow READS them and WRITES them, and the
    permitted transitions are a declared artefact (a 4-flag transition table),
    but the flow does not OWN them -- it cannot, because they change between
    runs from facts the flow never sees (a payment arrives; a probation clock
    expires at midnight). So they are input columns and output columns with a
    declared transition check between, which is the only shape that both
    replays and stays correct when nobody ran anything.

    "Collapsing any two of them into one field is a defect that will be found at
     a reporting date, by Finance, under time pressure."
    """
    pass  # load four flags + clocks; assert the transition is permitted


Relationship = module(subject_kind, facility_decision_history, existing_state,
                      name="relationship", owner="business_origination_engineering",
                      taps=["subject_kind", "review_basis_code", "branch_path"])
