"""The 04:15 run. 2.3M accounts, decided and allocated, published by 05:45.

Read top to bottom: this file is the whole flow, and the order in it is the
order of execution. Nothing here contains logic — doc 07 §2's lint holds, every
`def` below appears in the expression.

THE ORDER IS LOAD-BEARING IN ONE PLACE AND THE SPEC SAYS WHY.

  matrix -> overlays -> sequence -> SUSPENSION GATE -> allocation

Spec §5.4: "Where a commercial overlay and a suspension would disagree, the
suspension wins BY CONSTRUCTION rather than by evaluation order, and the evidence
must show that the suspension was evaluated against the OVERLAID recommendation,
not against the raw one."

Those two sentences pull in opposite directions and both are satisfiable. The
gate is downstream of the overlay, so the evidence shows the overlaid
recommendation being refused. And the suspension wins by construction because a
suspension is not evaluated against the recommendation at all — the Panel
computes a permitted set from the account's status, independently, and the Gate
intersects. There is no ordering in which a commercial overlay could win,
because there is no expression in which they meet as peers.
"""

import polars as pl

from decider2 import Gate, Runtime, fuse, parallel, pipeline
from decider2.frame import Join

from ..allocation.constraints import CapacityAllocation
from ..allocation.ranking import Ranking
from ..cohorts.assignment import Cohorts
from ..evidence.record import AssignmentRecord, SuspensionAttestation  # project 09
from ..matrix.dimensions import band_edge_flag, cell_id
from ..matrix.grid import lookup_treatment
from ..overlays.guard import CollectionsOverlays
from ..path.sequence import (
    CollectionsPath, cooling_off_remaining, expected_next, test_intervals,
)
from ..settlement.justification import SettlementAssessment
from ..state.assembly import AssembleState, bureau_state_code
from ..state.episode import Episodes
from ..suspensions.notices import notice_period_expires_on
from ..suspensions.panel import SuspensionPanel
from ..suspensions.prescription import (
    days_to_prescription, prescription_date, prescription_period_days,
    pre_prescription_window,
)
from ..scoring import CollectionsScorecard, RecoveryCurves, ContactBanding  # core.scorecard

# ---------------------------------------------------------------------------
# 1. ASSEMBLY — the one mandatory frame-tier crossing.
#    180M contact rows + 27.6M payments + 55.2M rolls -> ~190 scalars/account.
#    Every window in timelines/windows.py rides in this, grouped into one pass
#    per timeline. Measured shape: 6 passes, 11 minutes of the 90.
# ---------------------------------------------------------------------------

Assembly = AssembleState | Episodes | bureau_state_code

# ---------------------------------------------------------------------------
# 2. RISK — scored for everyone, including accounts where no treatment is
#    permitted, because the estimates are needed for RANKING even where nothing
#    may be done (spec §5.3 precondition). Suspensions do not gate this stage.
# ---------------------------------------------------------------------------

Risk = fuse(
    CollectionsScorecard          # 28 characteristics; null is a bin
    | CollectionsOverlays.on("collections.score")     # score_shift, odds_multiplier
    | ContactBanding
    | CollectionsOverlays.on("collections.band")      # band_edge_shift
    | RecoveryCurves                                   # recovery_estimate, cost_to_collect
)
# fuse() here and nowhere else in this pipeline: five modules of arithmetic-heavy
# work with no branching, which is the shape doc 02 §1.1 measures as fusion-
# positive. Everything downstream branches hard and is left unfused. The choice
# is authored, cannot change an answer, and is part of the structure fingerprint.

# ---------------------------------------------------------------------------
# 3. MATRIX — cell id, lookup, then the matrix overlay stack.
# ---------------------------------------------------------------------------

Matrix = (
    Cohorts
    | cell_id
    | band_edge_flag
    | lookup_treatment
    | CollectionsOverlays.on("matrix.intensity", "matrix.retries",
                             "matrix.cooling_off", "matrix.treatment_availability")
)

# ---------------------------------------------------------------------------
# 4. PATH — what have we already done, and what does that make today's action.
#    Reads the matrix's recommendation AND the account's treatment history.
# ---------------------------------------------------------------------------

Path = (
    CollectionsPath
    | test_intervals
    | cooling_off_remaining
    | expected_next
)

# ---------------------------------------------------------------------------
# 5. SUSPENSIONS — complete evaluation, then the gate.
#    prescription_date must be computed BEFORE the panel, because suspension 114
#    reads it. It is recomputed every day for every account and never stored.
# ---------------------------------------------------------------------------

Prescription = (
    prescription_period_days | prescription_date | days_to_prescription
    | pre_prescription_window
)

Suspensions = (
    Prescription
    | notice_period_expires_on
    | SuspensionPanel
)

Refusal = Gate(
    permitted="permitted_treatment_mask",
    ceiling="intensity_ceiling_from_suspensions",
    proposes=("treatment_code", "treatment_intensity", "channel_code"),
    on_block="record",                 # non-selection code 210 with every code that applied
    on_downgrade="record",             # the downgraded-to treatment AND what it replaced
    emits=["gated_treatment_code", "gated_treatment_intensity",
           "gate_outcome_code", "blocked_by_suspension_mask"],
)

# ---------------------------------------------------------------------------
# 6. ALLOCATION — population-level. @population_dependent, declared.
# ---------------------------------------------------------------------------

Allocation = Ranking | CapacityAllocation

# ---------------------------------------------------------------------------
# 7. OUTPUT
# ---------------------------------------------------------------------------

daily = pipeline(
    name="collections_daily",
    stages=(
        Assembly
        | Risk
        | Matrix
        | Path
        | Suspensions
        | Refusal
        | SettlementAssessment
        | Allocation
        | AssignmentRecord
        | SuspensionAttestation
    ),
    realtime_compatible=True,
    # ^ asserts at BUILD time that every state-vector field has an incremental
    #   form and that no stage is population-dependent upstream of the record
    #   boundary the live-call path reuses. Allocation is population-dependent
    #   and is therefore excluded from the live path automatically rather than
    #   by a second pipeline definition that can drift. See pipelines/live_call.py.
)


def run(frame: pl.LazyFrame, params, shared, overlays, rt: Runtime):
    """decision_date and knowledge_cutoff both arrive in `shared`.

        shared = {"decision_date": date(2026, 9, 19),
                  "knowledge_cutoff": Timestamp("2026-09-19T04:15:00Z"),
                  "calendar_version": 11}

    A recovery run for a missed day (NFR "Recovery") passes the MISSED date and
    that day's watermark, and gets that day's answer — not today's rules against
    yesterday's book. There is no other switch to remember to set, because there
    is no other source of time anywhere in the project.
    """
    pass
