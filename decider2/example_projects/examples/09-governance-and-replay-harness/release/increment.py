"""The increment is the unit of governance. Not the release.

THE DEVIATION, AND IT IS THE LARGEST ONE IN THIS PROJECT
-------------------------------------------------------
doc 08 §5 gives `decider2.impact(active, candidate, sample) -> ImpactReport`:
two generations, one number. Spec §5.5 needs n+1 runs over one population with
each change applied cumulatively in a declared order, because a Credit
Committee asking "which one of these five cost us the 1 840 approvals?" cannot
be answered by comparing the ends.

You cannot decompose a release after the fact. `git diff` between two builds
gives you a pile of changes, not an ordered sequence of individually buildable
generations. So the decomposition has to be AUTHORED, and if it is authored it
has to be cheap, and if it is cheap it has to be the thing people already write.

Hence: a release IS a list of increments, and each increment is independently
materialisable into a pipeline generation via doc 08 §4's `stage`/`activate`.
The release manifest is not documentation of the change; it is the change.

    release = Release(
        base=Pin.from_build("flow03@2027-03-31"),
        flow="flow03.unsecured_granting",
        increments=[
            Increment.table("rate_card.flex", to="RC-FLEX-2027-04",
                            declares=Moves(population="grades 7-9 Flex volume",
                                           direction="rate up", magnitude="+0.31 pp mean")),
            Increment.param("cap_register.tenure_floor_months", 24, 18,
                            declares=Moves(population="tenure 18-24 months",
                                           direction="amount cap up", magnitude="~2 100/month")),
            Increment.interior("cap_register", adds=("CAP-0455",),
                            declares=Moves(population="enquiry velocity 8+/90d",
                                           direction="amount cap down", magnitude="~640/month")),
            Increment.overlay("ADJ-2026-114", action="expire",
                            declares=Moves(population="all applicants",
                                           direction="affordability capacity up",
                                           magnitude="buffer 15% -> 12%")),
            Increment.skeleton(build="flow03@2027-04-24",
                            declares=NoOutcomeChange(reason="extract the solve into "
                                                            "modules/solve/; no logic change")),
        ],
        order_rationale="rate card first because the threshold move is priced through it; "
                        "the overlay expiry last because it is the only one we can unwind "
                        "same-day if the swap set is wrong.",
    )

FOUR THINGS THIS BUYS, EACH OF WHICH IS A SPEC REQUIREMENT

* §5.5 attribution: n+1 runs, mechanically, with no manual re-staging.
* §5.6 declared expected effect: it is a REQUIRED field of the increment, so a
  release literally cannot be described without declaring one per change.
* §5.14.4 overlay attribution: an overlay change is an increment exactly like a
  table change. This is the answer to §13 Q19 - an overlay changes an answer
  with no code change and no artefact change, and every other mechanism sees
  nothing, UNLESS the release manifest is what drives the harness rather than
  the git diff. The adjustment register emits an increment automatically on
  every register change, so an overlay cannot be changed outside a release.
* §5.15.13 individually attributable changes, or the release is split: this is
  now checkable. `Increment.skeleton` with more than one logical change in it is
  a judgement call and nobody can check it - but a release with ONE skeleton
  increment covering four rule additions is visible as such in the artefact,
  and the Credit Committee can see which teams do it. Spec §5.5 says the choice
  "must be visible to the Credit Committee rather than discovered during an
  incident", and that is exactly what this achieves and all it achieves.
"""

from __future__ import annotations

from datetime import date
from typing import Literal


class Declaration:
    """Required on every increment. §5.6's declared expected effect."""


class Moves(Declaration):
    population: str
    direction: str
    magnitude: str
    golden_records_expected: tuple[str, ...] | None   # by id, where the team can name them


class NoOutcomeChange(Declaration):
    """A refactor, a rename, a reorganisation, a library upgrade.

    Spec §5.6's no-effect rule: a NoOutcomeChange increment that moves ANY
    golden output blocks the release. No tolerance band, no team-level override.
    The only remedies are to fix the change or to re-declare it as Moves and
    take it through impact review. See artefacts/certification-REL-2027-04-FLX.md,
    where this rule blocks increment 5 and is the reason the release is any good.
    """
    reason: str


class Increment:
    ordinal: int
    kind: Literal["table", "param", "interior", "overlay", "skeleton", "capability"]
    target: str
    before: object
    after: object
    declares: Declaration
    author: str
    approval_ref: str
    changes_fingerprint: bool     # skeleton and interior: yes. table, param, overlay: no.

    @classmethod
    def table(cls, family: str, *, to: str, declares: Declaration) -> "Increment": ...
    @classmethod
    def param(cls, path: str, before, after, *, declares: Declaration) -> "Increment": ...
    @classmethod
    def interior(cls, module: str, *, adds=(), removes=(), edits=(), declares: Declaration) -> "Increment": ...
    @classmethod
    def overlay(cls, overlay_id: str, *, action: Literal["add", "expire", "amend", "disable"],
                declares: Declaration) -> "Increment": ...
    @classmethod
    def skeleton(cls, *, build: str, declares: Declaration) -> "Increment": ...
    @classmethod
    def capability(cls, capability: str, *, to: str, declares: Declaration) -> "Increment": ...

    def materialise(self, base: "Generation") -> "Generation":
        """Apply this increment on top of `base` and return a runnable generation.

        table/param/overlay: doc 08 §4 `rt.params.swap()`, microseconds, no compile.
        interior:            `rt.stage()` then `activate()`, one background compile.
        skeleton:            a different image; the ordering constraint below.
        """
        pass


class Release:
    base: "Pin"
    flow: str
    increments: tuple[Increment, ...]
    order_rationale: str          # required prose; interactions exist and order is not commutative

    def check_orderable(self) -> None:
        """A skeleton increment cannot be interleaved with interior increments
        of modules it removes. The only real ordering constraint, and it is
        structural, so it is checked rather than reviewed."""
        pass
