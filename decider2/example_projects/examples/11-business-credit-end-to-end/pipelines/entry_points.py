"""The nine entry points. One flow, five execution profiles. Spec 5.1, 8.1, 13-Q20.

    "A design in which an entry point is a wrapper that calls 'the flow' and
     then post-processes will discover that four of the nine do not run most of
     the flow at all."

So an entry point is not a wrapper. It is the pipeline SCOPED BY the governance
matrix that already exists as a Credit Governance artefact:

    review_scope_matrix.csv     9 entry points x 28 phase-parts = 252 cells

`scoped_by` is the project's third invented construct and the one that most
directly answers 13-Q20 and 13-Q22 together. Three properties:

  S1  The matrix is the control flow. There is no Python list of phases per
      entry point to drift from it (lifecycle/amendment/reopen.py M1).
  S2  `assert_matrix_total` checks both directions on every build: every
      phase-part in the matrix exists, every phase-part in the pipeline is named
      in the matrix, no blanks, 252 cells. Zero orphans permitted.
  S3  It renders. Spec 9.4 requirement 3 -- "'what does the annual review
      actually do' must render as a coherent document of its own, drawn from the
      same source, showing phases run in full, in part and not at all" -- is
      `EP3.render()`, not a document somebody maintains.

What `scoped_by` costs, honestly: PARTIAL is the hard value. "O5, changed
entities only" and "O7, spreading of the delivered figures only" are not a
phase minus some steps -- they are a phase over a different subject set. So
PARTIAL carries a named part, the part is declared on the phase, and a matrix
cell naming a part the phase does not declare fails the build. 28 phase-parts
rather than 23 phases is exactly this: five phases have a second declared part.
"""

from decider2 import governance_matrix, assert_matrix_total, RunLevel, entry
from pipelines.origination import Origination
from pipelines.annual_review import AnnualReview
from pipelines.covenant_test import CovenantTesting
from pipelines.early_warning import EarlyWarning
from pipelines.cascade import GroupCascade
from pipelines.pre_assessment import PreAssessment
from lifecycle.amendment.reopen import amendment_reopen
from lifecycle.forbearance.classification import Forbearance

review_scope = governance_matrix(
    "tables/governance/review_scope_matrix.csv",
    rows="entry_point", cols="phase_part", values=RunLevel,
    owner="credit_governance", effective_dated=True, total=True,
)

# --------------------------------------------------------------------------
# The nine. Profile letters are spec 8.1's.
# --------------------------------------------------------------------------

EP1 = entry("EP-1", "New-to-bank business application", profile="A",
            volume="900/day", budget="p95 4s, p99 8s",
            pipeline=Origination.scoped_by(review_scope, "EP-1"))

EP2 = entry("EP-2", "Additional facility, existing client", profile="A",
            volume="1 400/day", budget="p95 4s",
            pipeline=Origination.scoped_by(review_scope, "EP-2"))

EP3 = entry("EP-3", "Annual review", profile="C",
            volume="180 000/year in monthly cohorts of ~15 000",
            budget="8 hours overnight",
            pipeline=Origination.scoped_by(review_scope, "EP-3") | AnnualReview,
            pins_versions_at="batch_start")     # spec 5.15.2 req 2, 8.2

EP4 = entry("EP-4", "Covenant test", profile="D",
            volume="2.4 M/year; peak month 310 000",
            budget="nightly inside 4h; certificate-driven within 2 business days",
            pipeline=CovenantTesting)           # runs almost none of O1-O17

EP5 = entry("EP-5", "Early warning evaluation", profile="E",
            volume="240 000 facilities/day", budget="3h, complete by 05:00",
            pipeline=EarlyWarning)

EP6 = entry("EP-6", "Limit or facility amendment", profile="A",
            volume="600/day", budget="p95 4s",
            # The ONLY entry point whose scope is not a fixed matrix row: it is
            # the amendment re-open matrix row for the amendment's own kind,
            # resolved per record. A second governance matrix, same construct.
            pipeline=Origination.scoped_by(amendment_reopen, row="amendment_kind"))

EP7 = entry("EP-7", "Restructure or forbearance", profile="A",
            volume="120/day", budget="p95 20s -- a search, not a lookup",
            pipeline=Origination.scoped_by(review_scope, "EP-7") | Forbearance)

EP8 = entry("EP-8", "Group exposure re-assessment", profile="E",
            volume="~1 900 cascades/day touching ~14 000 facilities",
            budget="p95 15 min from the triggering event; bounded",
            pipeline=GroupCascade)

EP9 = entry("EP-9", "RM pre-assessment", profile="B",
            volume="3 000/day", budget="p95 under 2s, NO EXTERNAL CALLS",
            pipeline=PreAssessment,
            writes_decision_of_record=False)    # structurally. See pre_assessment.py

ENTRY_POINTS = [EP1, EP2, EP3, EP4, EP5, EP6, EP7, EP8, EP9]

# --------------------------------------------------------------------------
# Checks. All static.
# --------------------------------------------------------------------------

SCOPE_CHECK = assert_matrix_total(review_scope, Origination)

# Spec 5.15.2 requirement 3: every change declares which entry points it
# affects, and a change affecting more than three requires a joint release with
# a named coordinator. Computed from lineage, not declared by the author --
# which means it cannot be under-declared by someone who did not realise O11
# runs everywhere.
BLAST_RADIUS = entry.blast_radius(ENTRY_POINTS, threshold=3,
                                  on_exceed="require joint release + coordinator")

# Spec 5.1 EP-9 requirement 4, and spec 8.3 tension 1: where an EP-9 answer and
# a later EP-1 or EP-2 answer differ by more than the stated confidence band,
# the difference must be attributable to named causes -- data acquired since,
# entities assessed since, a policy change, or the pre-assessment's own declared
# assumptions proving wrong. An unattributable divergence is a DEFECT.
#
# This is the same machinery as the six-way cause decomposition (spec 5.4.1),
# pointed at a different pair. Reusing it is what stops EP-9 diverging from EP-1
# within two quarters, which spec 8.3 says is what happens if EP-9 is a separate
# path.
RECONCILIATION = entry.assert_reconcilable(EP9, against=[EP1, EP2],
                                           causes=["data_acquired", "entities_assessed",
                                                   "policy_change", "assumption_failed"],
                                           band="confidence_band")
