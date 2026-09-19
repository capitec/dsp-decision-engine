"""The monthly increase programme. 4.1 M accounts, three hours, one expression.

Read top to bottom this is the whole of s5. The only things that are not
visible here are values, which live in artefacts and params documents, and
that is the point of the split.
"""

from decider2 import Partition, fuse, parallel, pipeline
from decider2.frame import Sort, Union

from clm.affordability.assess import Assess
from clm.affordability.evidence import ExpenseBasis, IncomeEvidence, Obligations
from clm.allocation.fairness import ConditionalPass, FairnessReserve
from clm.allocation.ranking import Ranking
from clm.allocation.sweep import Allocation
from clm.caps.panel import CapChain
from clm.decrease.precedence import AttachClientState, Precedence
from clm.eligibility.panel import Eligibility, Exclusions
from clm.features.banding import Banding
from clm.features.temporal import Temporal
from clm.matrix.assignment import Matrix
from clm.offer.construct import ApplyIncrease, Construct
from clm.scoring.behavioural import Scoring
from clm.sources.book import Book
from clm.vocabulary import vocabulary

# --- the per-account core, shared with the real-time path ------------------
# This object, not a copy of it, is what pipelines/request.py composes. s5.12's
# agreement criterion is therefore a statement about evidence rather than about
# code: there is one expression of the matrix, the caps, the affordability
# logic and the overlays, and `assert_same_object` in tests/test_agreement.py
# is a one-line test because it is checking identity, not equality.

ProposedLimit = (
    Banding
    | Temporal
    | parallel(Scoring)              # 2.67 M rows, 32 characteristics: worth prange
    | Matrix
    | IncomeEvidence
    | ExpenseBasis
    | Obligations
    | Assess
    | fuse(CapChain)                 # seven cheap caps and a rounding: one kernel
)

# --- the monthly cycle -----------------------------------------------------

programme = pipeline(
    Book
    | Exclusions                     # all sixteen, complete attribution, ~40 ms
    | Eligibility
    # Work avoidance is frame tier. 1.43 M excluded accounts skip scoring, the
    # bureau join and affordability entirely, and rejoin with their exclusion
    # codes intact so all 4.1 M get a record (s7).
    | Partition(
        on="is_considered",
        taken=ProposedLimit | Ranking,
        skipped="passthrough",
        rejoin=Union(schema="schemas/cycle_record.json"),
    )
    # The decrease path has already run today (pipelines/decrease_daily.py);
    # its client-level roll-up suppresses increases here.
    | AttachClientState
    | Precedence
    | Construct                      # minimum meaningful increase, BEFORE ranking
    | FairnessReserve                # 15% of the envelope, by segment
    | Allocation                     # Sort | Sweep -- the population stage
    | ConditionalPass                # the 213 000 on the conditional path
    | ApplyIncrease
).with_vocabulary(vocabulary)


# --- running it ------------------------------------------------------------
#
#   from decider2 import artefacts
#   from datetime import date
#
#   live = artefacts.resolve(decision_date=date(2026, 9, 1), selector="live")
#   #  -> raises OverlayLapsed if any overlay in force is past its review date.
#   #  -> raises ArtefactMissing if the ALCO instruction is not this cycle's.
#
#   out = programme.apply(
#       book,
#       params=params,                       # ALCO instruction + policy values
#       shared={"decision_date": date(2026, 9, 1), "ranking_objective_code": 1},
#       tables=live,                         # THE ARTEFACT SET, as an argument
#       origin=f"cycle:2026-09/{live.fingerprint()}",
#   )
#
# The artefact set is an argument. That single choice is what makes simulation
# one implementation rather than two: see clm/simulate.py.

# --- restartability (s8) ---------------------------------------------------
# The three-hour window has no room for a full restart. Each `|` boundary is a
# checkpointable stage keyed on (snapshot_id, structure_fingerprint,
# artefact_set_fingerprint, params_digest, stage_name), so a resumed cycle
# reads the last complete stage and produces the same result as an
# uninterrupted one. The Sweep is a single indivisible stage, which is fine
# because it runs in seconds over 708 000 rows.
CHECKPOINT = "stage"
