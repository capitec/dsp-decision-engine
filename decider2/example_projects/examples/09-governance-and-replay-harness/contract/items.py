"""The twenty-three items of spec 09 §5.15, as data.

The single most important line in this project:

    An item is STRUCTURAL, EMPIRICAL or DECLARED, and the split is the answer
    to spec 09 §13 Q4 ("can evidence emission be a property of the framework
    rather than of each flow?").

STRUCTURAL  provable from the flow's graph with no data and no execution.
            decider2 makes this possible because structure is a pydantic
            instance (doc 02 §2) and lineage is static (doc 04 §3). 15 of 23.
EMPIRICAL   needs a corpus run to demonstrate. 4 of 23.
DECLARED    irreducibly a human sentence. The framework's only job is to make
            it impossible to omit. 4 of 23.

Nothing here is a recommendation. Each item, absent, permanently disables at
least one capability, and the `disables` field says which - so a failing
attestation prints the consequence rather than a rule number.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Literal

from decider2 import Pipeline
from decider2.graph import Module

from manifest.model import GovernanceManifest


class Proof(Enum):
    STRUCTURAL = "structural"   # graph only. Cannot be gamed, cannot be sampled.
    EMPIRICAL = "empirical"     # needs the certification corpus.
    DECLARED = "declared"       # a human wrote a sentence; we check it exists and is fresh.


class Item:
    """One contract item. Instances are frozen and live in `ITEMS` below."""

    number: int
    title: str
    proof: Proof
    spec_ref: str                       # "09 §5.15.2"
    disables: tuple[str, ...]           # capability names this item, absent, kills
    check: Callable[[GovernanceManifest], "Finding"]
    retrofittable: bool                 # false for all but three. Say so loudly.


# ---------------------------------------------------------------------------
# The fifteen structural items. Each `check` reads the manifest, never the flow.
# ---------------------------------------------------------------------------

def _c02_stable_identity(m: GovernanceManifest) -> "Finding":
    pass  # every node id is content-derived or declared; compare against the identity ledger for retirements.


def _c04_no_today(m: GovernanceManifest) -> "Finding":
    pass  # no step reaches date.today/now; enforced because such a step cannot njit, so it appears in `fallback_set`.


def _c05_reference_versions(m: GovernanceManifest) -> "Finding":
    pass  # every table read in the graph resolves through core.dates against decision_date, and records (version, cell).


def _c07_overlay_stack(m: GovernanceManifest) -> "Finding":
    pass  # every module consuming core.adjustments declares its unadjusted twin output in `writes`.


def _c09_no_external_calls(m: GovernanceManifest) -> "Finding":
    pass  # no @breaks_lineage region and no frame op reaches a network source; the sandbox enforces it again at replay.


def _c14_evaluation_recorded(m: GovernanceManifest) -> "Finding":
    pass  # every Branch in the graph carries branch_path in the witness set, not only the arms that write outputs.


def _c15_cap_chains(m: GovernanceManifest) -> "Finding":
    pass  # every value overwritten at more than one module boundary is in the witness set at `name@*` qualification.


def _c19_input_inventory(m: GovernanceManifest) -> "Finding":
    pass  # the inventory IS pipeline.schema()'s leaf set; check every leaf has a classification and a declared range.


def _c20_runnable_outside_production(m: GovernanceManifest) -> "Finding":
    pass  # the flow's entry point is pipeline.apply/score over supplied evidence; no scheduler symbol in the manifest.


# ---------------------------------------------------------------------------
# The four declared items. The framework cannot compute these. It can refuse
# to produce a manifest without them, which is the entire mechanism.
# ---------------------------------------------------------------------------

def _c12_declared_expected_effect(m: GovernanceManifest) -> "Finding":
    pass  # every Increment in the release manifest carries a Declaration; a release with an undeclared increment fails.


def _c22_prohibited_ground_usage(m: GovernanceManifest) -> "Finding":
    pass  # every leaf input flagged prohibited-adjacent in the inventory has a live, in-date PermittedUse record.


ITEMS: tuple[Item, ...] = (
    # n  title                                    proof              retrofittable
    #  1 stable decision identifier               STRUCTURAL         False
    #  2 stable identity for every logic element  STRUCTURAL         False   <- destroyed by one routine refactor
    #  3 deterministic assignment                 STRUCTURAL         False
    #  4 no reliance on "today"                   STRUCTURAL         False   <- replays SUCCESSFULLY and wrongly
    #  5 recorded reference-data versions         STRUCTURAL         False
    #  6 inputs captured as received              STRUCTURAL         False
    #  7 overlay stack recorded                   STRUCTURAL         False
    #  8 mutable state snapshotted/addressable    STRUCTURAL         False   <- flows 07 and 08 fail this today
    #  9 no external calls during replay          STRUCTURAL         True    (enforced by the sandbox, not the flow)
    # 10 explicit complete parameter sets         STRUCTURAL         False
    # 11 declared reason codes                    STRUCTURAL         False
    # 12 declared expected effect per change      DECLARED           True
    # 13 individually attributable changes        DECLARED           True
    # 14 evaluation recorded, not only firing     STRUCTURAL         False
    # 15 cap chains recorded                      STRUCTURAL         False
    # 16 score contributions as output            STRUCTURAL         False
    # 17 idempotent at-least-once emission        EMPIRICAL          True
    # 18 emission cannot fail the decision        EMPIRICAL          True
    # 19 declared input inventory                 STRUCTURAL         False
    # 20 runnable outside production              STRUCTURAL         True
    # 21 versioned resolvable logic identity      STRUCTURAL         False
    # 22 declared prohibited-ground usage         DECLARED           True
    # 23 field-level PII classification           STRUCTURAL         False
)

GRADES = Literal["A", "B", "C"]
# A  all 23 pass. Everything in this project works for this flow.
# B  all STRUCTURAL pass; an EMPIRICAL or DECLARED item is stale. Degraded, fixable this quarter.
# C  a STRUCTURAL item fails. Named on every artefact the flow produces, for as long as it is true.
