"""The live overlay register, across all eight flows, answerable as at any past
date.

Overlays are governed here rather than in the flows for the reason spec §5.14
states and that no framework doc addresses: an overlay changes answers with NO
code change, NO structure change and NO table change - so version diff,
certification, coverage and the reviewable artefact all report nothing while
outcomes move. An estate with overlays and without this capability is an estate
where the most frequently changed thing is the least governed.

WHERE THE REGISTER LIVES, AND WHY IT IS HERE AND NOT IN 00
  00 §6.22 publishes `core.adjustments` as a capability: it APPLIES overlays.
  The register - which overlays exist, over what scope, in what order, approved
  by whom, expiring when - is a versioned artefact in the harness's own store,
  effective-dated like a rate card, and each flow's build pins the register
  version it read. Two consequences:
    - "which overlays were in force on 14 September 2026, over what scope, in
      what order" is a point-in-time query over one artefact, not an
      archaeology exercise across six flows;
    - a register change EMITS A RELEASE INCREMENT automatically
      (release/increment.Increment.overlay), which is how an overlay becomes
      visible to swap-set attribution. This is the answer to §13 Q19 and it is
      the only one that works: you cannot make a git-driven mechanism see a
      change that is not in git, so the harness must be driven by the release
      manifest and the register must be one of its sources.

SCOPE, AND SPEC §13 Q21
  "What is the unit of comparison for overlay scope?" An overlay scoped to
  grades 9-12 on channel 4 for product 10 is not comparable cell-for-cell with
  one scoped to a sector code in flow 05. Flattening them into a common
  coordinate space produces a number that means nothing.

  So the register does NOT flatten. Scope is stored as a predicate over the
  canonical vocabulary (00 §4), and the two aggregate questions the committee
  asks are answered by EVALUATING those predicates over a population rather
  than by comparing them to each other:
     "how much of the estate is currently overlaid" -> the share of last
        month's decisions on which at least one overlay fired;
     "where do overlays concentrate"                -> that share, grouped by
        product, grade and channel.
  Both are frame-tier aggregations over evidence that already exists (item 7
  records the overlay stack on every decision). Nothing needs to be
  commensurable; the population makes them commensurable.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import polars as pl


class Overlay:
    overlay_id: str                    # ADJ-2026-114
    kind: Literal["score_shift", "scaling_change", "odds_multiplier", "calibration_reanchor",
                  "grade_boundary_shift", "cutoff_shift", "rate_addon", "cap_reduction",
                  "buffer_adjustment", "matrix_intensity"]
    flow: str
    scope: "ScopePredicate"            # a predicate over 00 §4 names, not a flattened key
    magnitude: "Magnitude"             # points | ratio | pp | bp, with its unit, never a bare float
    position: int                      # composition order is declared, not emergent (00 §6.22.3)
    owner: str
    approval_ref: str                  # a committee minute, not a ticket
    rationale: str                     # required field, not a courtesy
    effective_from: date
    effective_to: date
    review_date: date
    enabled: bool                      # so the stack runs off without deleting the definition
    supersedes: str | None
    renewals: tuple["Renewal", ...]


def as_at(d: date) -> tuple[Overlay, ...]:
    """The register in force on any past date, across all eight flows, with
    order. Asked by a replay and by a regulator in the same words."""
    pass


def concentration(month: str) -> pl.LazyFrame:
    """Share of decisions with >=1 overlay fired, by product x grade x channel.
    Frame tier over the recorded overlay stacks."""
    pass


def estate_coverage(month: str) -> float:
    """One number for the committee: what fraction of the estate's decisions
    were touched by at least one overlay. Currently ~31%, which is the number
    that starts the conversation."""
    pass


def changes(since: date) -> tuple["RegisterChange", ...]:
    """Every change to the register, with before and after, author and approval
    (spec §5.14 'what must be recorded'). Each one is also an Increment."""
    pass


# ---------------------------------------------------------------------------
# Ageing and expiry - the reports that make "mandatory review date" mean
# something. Rendered monthly; see artefacts/adjustment-register-2027-03-31.md.
#
# Spec 09 §5.14.2's six reports, and the note that the last one is the only one
# with teeth: an expiry report that cannot say what unwinding would cost
# produces indefinite renewal, which is indistinguishable from having no
# expiry at all. `unwind_estimate` lives in adjustments/unwind.py because it
# needs the execution plane; everything below is a query over the register.
# ---------------------------------------------------------------------------
def past_review_date(as_at: date) -> pl.DataFrame:
    """By age. The dominant failure mode: a tightening applied in one bad
    quarter, still in force four years later. Sorted by days overdue
    descending, because the top row is the story."""
    pass


def renewed_without_rejustification(as_at: date, *, times: int = 2) -> pl.DataFrame:
    """Renewal as a formality rather than a decision. Detected by comparing the
    rationale text across renewals: an unchanged rationale on a third renewal
    is a formality by definition."""
    pass


def orphaned(as_at: date) -> pl.DataFrame:
    """Overlays whose approving committee member or owning team no longer
    exists. Joined against the HR feed and the org structure. Nobody left who
    can say why - which is the state that makes unwinding unthinkable."""
    pass


def never_fired(as_at: date, *, window_days: int = 90) -> pl.DataFrame:
    """Scoped to a population that no longer occurs. Dead logic in the §5.8
    sense, and measurable the same way, because item 7 records the stack on
    every decision including the overlays that were in scope and did not move
    a value."""
    pass


def dominant(as_at: date, *, threshold: float = 0.40) -> pl.DataFrame:
    """Firing on more than 40% of a flow's volume. No longer an adjustment - it
    is the policy, under the wrong approval, outside the artefact that governs
    it (spec §11.16).

    The report's output is a MIGRATION PLAN, not a flag: the base artefact it
    belongs in, the cells or boundaries it would become, and - the hard part -
    the mapping that keeps two years of decisions comparable after the move.
    See adjustments/unwind.py:absorb().
    """
    pass
