"""Overlay stack resolution — spec §5.3.4, and the answer to §13 Q16.

*"What is an overlay, structurally?  It is a parameter change, a piece of real
logic, and an audit artefact at the same time.  Is it one thing or three?"*

One artefact with three faces, and the resolution below is where the three are
separated without being collapsed:

    overlay register document   -->   (a) a ParamsDelta          values
                                      (b) a set of application    structure,
                                          SITES it must land on   declared in
                                                                  the skeleton
                                      (c) an overlay_stack_id     audit

(a) is free at run time.  (b) is code, reviewed in git, and the reason an overlay
kind nobody declared a site for cannot be applied silently.  (c) is what goes
beside the path, never into it.

RESOLUTION IS BY `cycle_date`, NEVER BY TODAY
----------------------------------------------
`resolve(as_at=cycle_date)` — a re-run of the March cycle in 2028 resolves
March's overlays in March's order, including the ones that have since expired.
Doc 08 §6.2's `resolve_params(doc, origin=, complete=)` has no temporal axis at
all; DEMANDS #5 asks for `as_at=` on every document resolution, because effective
dating is not a table concern (00 §6.19) — it is a property of every versioned
artefact in this project, and half of them are not tables.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel


class Overlay(BaseModel):
    overlay_id: str                     # "OV-2026-114"
    kind: Literal["cut_off_shift", "threshold_shift", "score_shift",
                  "odds_multiplier", "cap_reduction"]
    description: str
    owner: str
    rationale: str
    approval_ref: str
    approved_by: list[str]
    effective_from: date
    effective_to: date
    review_date: date                   # MANDATORY.  Default max 3 cycles for a
                                        # volume dial, 6 months for a risk overlay.
    scope: dict                         # campaigns / segments / channels / products /
                                        # grade ranges / tiers — declared, enforced
    target: dict                        # kind-specific; threshold_shift names a SLOT
    precedence: int                     # from registries/stacking_order.json


class ResolvedStack(BaseModel):
    overlay_stack_id: int               # the ordered set, hashed; recorded on every row
    cycle_date: date
    campaign_id: int
    ordered: list[Overlay]
    params_delta: dict                  # slot_id -> new value, tier -> new cap, etc.
    site_bindings: dict                 # site name -> params for that site's module
    unmatched: list[str]                # overlays whose scope matched nothing -> FATAL


def resolve(registry_doc: dict, *, as_at: date, campaign_id: int,
            tree: "TreeDocument", stacking_order: dict) -> ResolvedStack:
    """Resolve the stack in force, in declared order, and fail the cycle rather
    than warn.

    Four failure modes, all of them `raise`, all of them naming the overlay
    (spec §5.3.4 reqs 5 and 6, acceptance criterion 17):

      1.  `review_date < as_at` and no renewal recorded  ->  EXPIRED.
          The failure mode this prevents is a tightening approved for one bad
          quarter, still suppressing 200 000 clients a month three years later,
          which nobody can now explain (scenario 17).
      2.  scope names a campaign/tier/segment that does not exist at `as_at`
          ->  OUT_OF_SCOPE.
      3.  scope matches **nothing** -> MATCHES_NOTHING.  A tightening scoped to a
          campaign retired last month is caught, not assumed to be working.
      4.  a `threshold_shift` whose `target.slot` is not in
          `tree.slot_index`  ->  DANGLING_TARGET, naming the old node key and the
          identity map entry that explains where it went.  This is scenario 18 —
          a volume dial and a re-fit landing in the same cycle — and it is
          impossible to get wrong because the slot id contains the node key.

    Approval separation (req 4) is checked here too: a risk overlay
    (cut_off_shift / score_shift / odds_multiplier) with no Credit Risk approver,
    or a volume dial with no campaign forum approver, is rejected.  Neither may
    approve the other's kind, and the check is on the *kind*, so a volume dial
    relabelled as a risk overlay to get it past the forum fails on its target
    instead.
    """
    pass


def stack_id(ordered: list[Overlay]) -> int:
    """A stable int32 over (overlay_id, effective_from, precedence) in order.

    Recorded on every assignment and every path row.  Two cycles whose
    `overlay_stack_id` differs are **not comparable figures** and the reporting
    layer refuses to put them on one axis without a decomposition (spec §5.3.4(a),
    acceptance criterion 18).
    """
    pass


def windows_broken_by(overlay: Overlay, measurements) -> list[str]:
    """Spec §5.3.4(b): an overlay affecting a campaign with an in-flight
    measurement window may only take effect at a measurement boundary, unless the
    campaign owner and Campaign Analytics jointly accept the break and the
    acceptance is recorded against the measurement.

    Returns the windows this overlay's start/change/expiry falls inside.  The
    cycle refuses to run when one is returned without a recorded acceptance.
    "List, for any measurement window, every overlay that started, changed or
    expired inside it" is the same query read the other way.
    """
    pass
