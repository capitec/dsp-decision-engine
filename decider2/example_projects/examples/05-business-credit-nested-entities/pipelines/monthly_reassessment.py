"""The monthly re-assessment batch, and partial re-runs.

Two workloads over the same definition:

  * 180 000 existing business clients, ~1 120 000 entities and ~6 700 000
    events, in an 8-hour window (s8);
  * a single entity's bureau refresh, re-derived and re-rolled-up in under
    500 ms, "without re-running everything -- and the difference explained"
    (H5, s8, s10 acceptance 11).

--------------------------------------------------------------------------
WHY THE BATCH IS NOT A SEPARATE PIPELINE
--------------------------------------------------------------------------
s13 Q16: "Can the same assessment run record-at-a-time for granting and
set-at-a-time for the monthly batch and produce identical results, given that
the batch's 6 700 000 events are the same shape as one application's 600?"

With grains, that question dissolves: **there is no record-at-a-time mode in
the nested section**. The Event frame for one application has 600 rows; for the
batch it has 6.7 M. Same kernels, same folds, different row counts. The granting
path differs only in that `score()` marshals capacity-bounded arrays instead of
polars frames, which is a boundary difference and not a semantic one -- and the
equivalence ladder is what proves that, once extended to the fold
(FRAMEWORK-DEMANDS D06).

This file is therefore short. It is the same pipeline with pricing removed
(a re-assessment does not re-price) and two things added.
"""

from decider2 import Runtime, Each, Gather, invalidates
from grains import Application, Entity, Event
from pipelines.business_facility import (
    _structure, _business, _nested, _people, _grade,
)
from validation.consistency import FinalValidation

monthly_reassessment = (
    _structure | _business | _nested | _people | _grade | FinalValidation
).without(stage="pricing")


# --------------------------------------------------------------------------
# PARTIAL RE-RUN
#
# s13 Q8: "What does a partial re-derivation look like? One entity refreshes.
# Its events reclassify, its verdict changes, the blend weights are unchanged
# but the blend result is not, and the price may move. What is the smallest
# re-run that is provably equivalent to a full one, and how is that proof
# obtained?"
#
# The grain DAG answers both halves. It is a static dependency graph over
# stages, so the invalidation set of "the Event rows of entity 7 changed" is
# derivable without running anything:
#
#   Each(Event, ClassifyEvent)        -> only entity 7's event rows
#   Gather(Event -> Entity)           -> only entity 7's row
#   Each(Entity, Verdict | Scoring)   -> only entity 7's row
#   Gather(Entity -> Application)     -> ALL of it, because a fold reads every
#                                        child; but the other 39 entity rows are
#                                        unchanged and are read, not recomputed
#   everything at Application grain   -> recomputed
#
# The proof is the same property that gives ordering independence: every fold is
# commutative-associative and every step is pure, so recomputing a fold over an
# unchanged child frame plus one changed row gives the value a full re-run would.
# It is a property of the fold vocabulary, not of this project's care.
#
# The cost: one entity's events (median 6, max 60) through two kernels, one
# 40-row fold set, and the Application-grain tail. Nowhere near 500 ms.
# --------------------------------------------------------------------------
def refresh_one_entity(rt: Runtime, prior, entity_keys: list[str]):
    """Re-derive one entity and re-roll-up the business."""
    return rt.reapply(
        monthly_reassessment,
        prior=prior,
        changed=Entity.identity(entity_keys),
        explain=True,          # s9.5: the difference must be explainable at
                               # EVERY level -- which inputs changed, which
                               # classifications changed, which verdicts changed,
                               # which weights changed, which grade changed, and
                               # what that did to the offer.
    )

# `explain=True` returns a level-by-level diff, which is possible because both
# runs recorded the same declared values at the same grains with the same
# identities. s9.5: "A re-run that produces a different answer without an
# attributable cause is a defect." With witnesses recorded at every fold, "no
# attributable cause" is detectable rather than merely suspected.


# --------------------------------------------------------------------------
# BATCH SHAPE AND THE FAN-OUT NFR
#
# s8: "The p99 must not be dominated by fan-out. A 40-entity application may
# take longer than a 1-entity one; it may not take forty times longer end to
# end, because the per-application fixed work dominates at the median."
#
# This falls out of grains-as-frames and is the strongest performance argument
# in the sketch. In a per-application loop, a 40-entity application costs 40x
# the nested work SERIALLY, inside one request. As frames, a 40-entity
# application contributes 40 rows to a frame that already has thousands, and the
# nested stages are amortised across the batch.
#
# For the 3-second realtime budget the arithmetic is:
#   Application-grain work   ~ fixed, one row
#   Entity-grain work        ~ 1..40 rows through ~6 kernels
#   Event-grain work         ~ 0..600 rows through 2 kernels
#   Candidate-grain work     ~ 200..600 rows through 1 kernel, prange
#
# The 40-entity/600-event worst case is ~1 240 kernel rows against a 1-entity
# case's ~205. Under 8 seconds (s8's worst-case budget) with room, and the ratio
# is ~6x rather than 40x because the candidate grain -- which dominates -- does
# not vary with entity fan-out at all.
# --------------------------------------------------------------------------
BATCH_PLAN = monthly_reassessment.plan(
    chunk_by=Application,
    chunk_rows=20_000,                 # ~125 000 entities, ~750 000 events
    parallel_grains=[Entity, Event],   # authored, not inferred (doc 02 s3.3)
)
