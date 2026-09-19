"""The Flex Loan granting and pricing skeleton. `product_code` 10.

This file is the **only** place execution order lives. Everything else in this
project is a module, a table, an interior document or an adjustment set. An
engineer changes this file under a release; nobody else can change it at all.

Read the pipeline expression at the bottom top-to-bottom and you have read the
flow. Eleven stages, five overlay points, one bounded search, one waterfall.
"""

from __future__ import annotations

from decider2 import Halt, Map, fuse, parallel
from decider2.collections import Collection
from decider2.overlay import overlay

from adjustments.points import (
    BOUNDARY_SHIFT,
    CAP_REDUCTION,
    ODDS_MULTIPLIER,
    RATE_ADD_ON,
    SCALING_CHANGE,
    SCORE_SHIFT,
)
from modules import affordability, bureau_quality, disclosure, eligibility, fraud_handoff
from modules.caps.register import CapWaterfall
from modules.offers import objectives, viability
from modules.scoring import calibration_and_grading as cg
from modules.scoring import scorecard, segment
from modules.solve.search import SolveOneTerm
from modules.validation.independent import FinalValidation
from pipelines.numerics import FLEX_NUMERICS
from pipelines.reasons import (
    ABSENT_AFFORDABILITY_FAIL,
    ABSENT_FRAUD_DECLINE,
    ABSENT_GATE_DECLINE,
    AFFORDABILITY_FAIL,
    FRAUD_DECLINE,
)

# ---------------------------------------------------------------------------
# The permitted term list.  Nine values, product-owned, weekly cadence.
#
# `Collection` is a fixed-capacity per-record collection: capacity is declared
# because the record tier has no heap.  `filtered_by` entries are modules that
# return a bool per element AND a reason code, so a term that disappears is
# never a term that silently was not tried (§5.9 "and that removal is recorded
# with its reason").
# ---------------------------------------------------------------------------

permitted_terms: Collection[int] = Collection(
    name="permitted_terms",
    capacity=9,
    source=table("permitted_term_list"),  # 4 segments x 9 terms, product-owned
    filtered_by=[
        viability.term_within_cap,        # term_months <= term_cap    -> reason 1450
        viability.term_in_segment_list,   # per-segment permitted list -> reason 1451
    ],
)


# ---------------------------------------------------------------------------
# The flow.
# ---------------------------------------------------------------------------

granting = (
    # -- 1. Intake and the 14 eligibility gates. -----------------------------
    # GateSet evaluates EVERY gate whose inputs are in hand, and never
    # short-circuits: a client who fails four gates is entitled to be told
    # about four gates (§5.1).  Gates whose inputs are not yet retrieved emit
    # NOT_EVALUATED with the reason they could not be evaluated -- a third
    # state, not a collapse into pass or fail.
    eligibility.Gates
    | Halt.when(
        "is_eligible",
        is_=False,
        reason="primary_reason_code",
        absent_because=ABSENT_GATE_DECLINE,
    )
    # -- 2. Consent, then the fraud handoff. ---------------------------------
    # Consent absence is a REFERRAL, never a decline, and proceeding without it
    # must be structurally impossible: `requires_consent=` on the bureau module
    # makes the bureau stage unreachable without a valid consent verdict, so
    # there is no code path that skips the check.
    | fraud_handoff.ConsentCheck
    | fraud_handoff.FraudDisposition          # 4 handling paths incl. the bypass
    | Halt.when(
        "fraud_verdict_code",
        is_=FRAUD_DECLINE,
        reason=1210,
        absent_because=ABSENT_FRAUD_DECLINE,
    )
    # -- 3. Bureau data quality. ---------------------------------------------
    # core.bureau normalises; this stage owns the verdict on the result, and
    # the negative requirement: a data-quality problem may never become a
    # decline.  DQ-2 and DQ-3 refer.
    | bureau_quality.DataQuality
    # -- 4. Segment, score, calibrate, grade -- with four overlay points. -----
    | segment.AssignSegment
    | scorecard.Score                         # 45 contributions, always emitted
    | overlay(SCORE_SHIFT)                    # (1) score      -> score
    | overlay(SCALING_CHANGE)                 # (2) a PARAM overlay, see below
    | cg.Calibrate
    | overlay(ODDS_MULTIPLIER)                # (3) pd         -> pd
    | cg.Grade
    | overlay(BOUNDARY_SHIFT)                 # (4) risk_grade -> risk_grade
    # The challenger runs on a deterministic 10% of `client_id`, scores,
    # calibrates and grades in full, decides nothing -- and carries NO overlay
    # points, because a challenger carrying the champion's overlays measures
    # the overlay rather than the challenger (§5.4.1).
    | scorecard.Challenger.when(scorecard.in_challenger_sample)
    # -- 5. The cap waterfall: 52 rules, 3 ceilings, 1 authorised raise. ------
    | CapWaterfall
    | overlay(CAP_REDUCTION)                  # (5) amount_cap, term_cap
    # -- 6. Affordability, consumed whole from project 02. -------------------
    | affordability.Consume                   # incl. its table versions, verbatim
    | Halt.when(
        "affordability_verdict_code",
        is_=AFFORDABILITY_FAIL,
        reason=1310,
        absent_because=ABSENT_AFFORDABILITY_FAIL,
    )
    # -- 7. The solve.  One bounded search per permitted term. ---------------
    # `parallel(...)` is legal here and only here: the Map body has no
    # cross-row reduction, which FLEX_NUMERICS asserts statically (a prange
    # region containing a float accumulation would be a build error, because
    # summation order would differ between the batch and realtime paths and
    # acceptance criterion 11 would fail silently).
    | parallel(
        Map(
            over=permitted_terms,
            as_="term_months",
            body=fuse(SolveOneTerm),          # one kernel: the probe is hot
            capacity=9,
            collect=SolveOneTerm.interface.outputs,
            absent_because="term_suppression_reason_code",
        )
    )
    # -- 8. Offer set construction. ------------------------------------------
    | viability.MinimumViableOffers           # 5 rules, all reasons recorded
    | viability.Deduplicate                   # equal amount, instalments within 2%
    | objectives.Recommend                    # 3 objectives, one Branch, param-routed
    # -- 9. Final validation, independent by construction. -------------------
    | FinalValidation
    # -- 10/11. Disclosure. --------------------------------------------------
    | disclosure.Quotation                    # 5 components that must sum to the total
    | disclosure.Reasons                      # ranked, client-facing, language of record
).with_numerics(FLEX_NUMERICS).named("flex_loan_granting")


# ---------------------------------------------------------------------------
# What is NOT here, deliberately.
#
# 1. No frame-tier node.  Not one.  The bureau account list, the enquiry list,
#    the public-record list, the obligations list and the in-flight application
#    list all arrive as `Ragged[...]` per-record collections with declared
#    capacities (see schemas/application_input.json), so every aggregation over
#    them is record-tier arithmetic that is bit-identical in `apply()` and in
#    `score()`.  A `Join` or an `Aggregate` anywhere in this pipeline would
#    mean the realtime path could not run the same pipeline object, and
#    acceptance criterion 11 ("zero differences over 100 000 records") would
#    become a claim about two pipelines rather than one.  See FRAMEWORK-DEMANDS
#    #24.
#
# 2. No `if batch:`.  Both entry points run this object:
#        granting.apply(frame, params=p, tables=t)     # 648 rec/s, 6h window
#        granting.score(**application, params=p, tables=t)   # p99 < 120 ms
#    `t` is a resolved table bundle (see below) and is the same object in both.
#
# 3. No date.  `decision_date` is an input column; `tables` cannot be resolved
#    without one; and a lint forbids `datetime.now`/`date.today` anywhere under
#    modules/ or pipelines/.  "Today" appears nowhere in this project.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Invocation.  Both bundles are resolved once per invocation, outside the
# kernel, and both are stamped into the audit record automatically.
# ---------------------------------------------------------------------------

def invoke_batch(frame, params_doc, origin: str, decision_date):
    """Monthly pre-assessment over 14.2 M clients. One `decision_date` per run."""
    pass  # resolve params + tables + adjustments as-of, then granting.apply(...)


def invoke_realtime(application: dict, runtime):
    """One application. Reads the generation pointer exactly once (doc 08 §4)."""
    pass  # runtime.generation().score(**application)
