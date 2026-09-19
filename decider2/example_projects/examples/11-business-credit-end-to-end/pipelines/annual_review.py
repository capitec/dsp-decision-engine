"""EP-3 -- L1, the annual review. 164 decision points, all written here.

15 000 facilities a month in an 8-hour overnight window. This is the pipeline
that decides whether the Bank still wants this facility on these terms, and what
has changed since it last asked.

A review re-runs most of the origination assessment. What makes it a distinct
phase rather than a re-run is that it produces FOUR OUTPUTS ORIGINATION DOES NOT
PRODUCE, and all four are comparisons.
"""

from decider2 import fuse, batch_pin
from lifecycle.annual_review.basis import REVIEW_BASIS, ReviewBasis
from lifecycle.annual_review.migration import GradeMigration
from lifecycle.exit.handoff import Exit
from consumed.p06_restructure import LimitDecision
from consumed.p05_origination import PricingSearch
from modules.covenants.setting import CovenantSetting
from modules.authority.routing import Authority

# --------------------------------------------------------------------------
# The basis is chosen FIRST and everything downstream runs `.under` it.
#
# This ordering is the whole of spec 5.4.3's structural argument. Five declared
# modes of one assessment, selected up front, rather than a complete path with
# four error branches hanging off it. 56% of reviews take a mode other than 1,
# and in a branch-off-the-happy-path design that is 56% of the book decided in
# code nobody reviews.
# --------------------------------------------------------------------------
AnnualReview = (
    ReviewBasis                                   # which of the five, and why
    | GradeMigration                              # output 1: three grades + six causes
    | PricingSearch.as_repricing()                # output 2: contractual vs indicated
    | LimitDecision                               # output 3: p07, revolving only
    | CovenantSetting.as_reset_proposal()         # covenant reset proposal
    | fuse(Exit)                                  # output 4: exit recommendation
    | Authority
).under(REVIEW_BASIS)

# --------------------------------------------------------------------------
# Output 2 deserves a note, because it is the one a design drops.
#
# Spec 5.4.1 output 2: THE MARGIN SHORTFALL MUST BE REPORTED WHETHER OR NOT IT
# CAN BE ACTED ON. Most facilities cannot be repriced mid-term -- a product 50
# term facility at a fixed margin is fixed -- and the temptation is to skip the
# calculation where the action is None.
#
#   "the aggregate of unactionable shortfalls across the book is a portfolio
#    fact the Credit Committee needs, and a review that silently drops it
#    produces a book that is systematically underpriced for reasons nobody
#    recorded."
#
# `as_repricing()` therefore computes all three numbers unconditionally and the
# ACTION is a separate output. Three numbers and an action, not an action that
# may or may not have numbers behind it.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Batch version pinning. Spec 5.15.2 req 2, 8.2.
#
#   "A version change during an 8-hour review batch either does not apply to
#    that batch or aborts and restarts it. It NEVER applies to part of it."
#
# Doc 08 4 property 1 already gives half of this: `apply` reads the generation
# pointer exactly once per invocation, so a swap cannot split one batch. What it
# does not give is the TABLE and COMPONENT versions, which are params and inputs
# rather than generations. `batch_pin` freezes all three -- generation, manifest
# and every dated_table resolution -- into one frozen bundle for the cohort, and
# records its id on all 15 000 decisions of record.
#
# The failure it prevents is the one doc 08 4 property 1 names in the current
# implementation: "two config objects touched at different moments in one batch
# can be on different versions -- half a run decided under one threshold, half
# under another, with nothing recording where the boundary fell."
# --------------------------------------------------------------------------
AnnualReview = batch_pin(AnnualReview, at="cohort_start",
                         covers=["generation", "component_manifest", "dated_tables"],
                         on_change="abort and restart the cohort")

# Spec 9.2 property 3, run monthly over the whole book rather than per review:
# no gap where a review was due and no decision of record exists; no broken
# predecessor link; no grade on a scale with no registry entry.
#
#   "A book with zero exceptions is a book where the check is not working."
#
# so the check reports a count and the count is expected to be non-zero.
COHERENCE = AnnualReview.assert_sequence_coherent(
    over="the whole book", cadence="monthly",
    checks=["no_missing_review", "no_broken_predecessor", "no_unregistered_scale"],
    expect_nonzero_exceptions=True,
)
