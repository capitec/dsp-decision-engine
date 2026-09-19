"""The node identity map — spec §5.4.3 req 4, §5.9 sign-off, §11.2, §11.18.

Produced on every publication, against the immediately preceding version of the
same (campaign, variant).  Retained indefinitely.  Seen by the campaign owner and
Credit Risk *before* sign-off, and it is the only artefact in this project whose
audience is unambiguously non-engineering.

Classification is mechanical because identity is content-derived:

    carried_forward   key present in both versions
    added             key only in the new version
    removed           key only in the old version
    changed           NOT a computed class.  See below.

"Changed" cannot be computed from keys alone — a moved threshold looks exactly
like a removal plus an addition.  It is recovered by matching on a **weaker**
digest and reporting the delta:

    structural_key = blake2b(sorted (feature_id, op) pairs of the condition)

Two nodes with the same structural key and different node keys are the same
*test shape* with different constants, so the map reports them as `changed` and
prints what moved.  Two nodes with different structural keys are genuinely an
add and a remove.  This is a heuristic and it is labelled as one in the artefact
(`match: "structural"`), because a heuristic that a human reviews is fine and a
heuristic that silently produces a comparable time series is not.
"""

from __future__ import annotations

from pydantic import BaseModel

from .document import TreeDocument


class MapEntry(BaseModel):
    classification: str          # carried_forward | added | removed | changed
    node_key_from: str | None
    node_key_to: str | None
    match: str                   # "exact" | "structural" | "declared"
    level_from: int | None
    level_to: int | None
    author_label: str
    delta: list[str] = []        # e.g. ["threshold discretionary_income 2200 -> 2600"]
    continues: str | None = None # a human continuity assertion; see below
    prior_cycle_volume: int | None = None   # so the reviewer sees what breaks


class IdentityMap(BaseModel):
    campaign_id: int
    from_version: int
    to_version: int
    generated_at: str
    canon_version: int
    entries: list[MapEntry]
    summary: dict                # counts per classification
    comparability_note: str      # which node time series break, in prose


def build_identity_map(old: TreeDocument, new: TreeDocument, *, volumes) -> IdentityMap:
    """Classify every node in both versions and attach last cycle's volume.

    `volumes` is not decoration.  A campaign owner asked to sign off on a re-fit
    needs to see that the node whose identity is breaking carried 412 000 clients
    last month, not that "14 nodes changed".
    """
    pass


def apply_continuity_assertions(imap: IdentityMap, assertions: dict) -> IdentityMap:
    """Spec §13 Q5, made a first-class artefact instead of an argument.

    *"Is `x >= 2200` in March the same node as `x >= 2200 and y in {1,2,4}` in
    April, where the second condition is always true for the population that
    reaches it?"*

    The engine's answer is no, always, and it will never say otherwise — identity
    is mechanical.  But the campaign owner may record a **continuity assertion**
    in the sign-off document:

        continuity:
          - from: n_a1f45e9c2b70d863
            to:   n_b7302ce8149fa65d
            basis: "threshold move only; population reaching the node unchanged"
            approved_by: [campaign_owner_11, credit_risk_policy]

    which does two things and no more: it appears in the identity map with
    `match: "declared"`, and it permits the reporting layer to chain the two node
    time series **with the break annotated on the chart**.  It never merges the
    keys, never rewrites a stored path, and never makes the two nodes one node.
    A chained series that nobody approved cannot be produced at all.
    """
    pass
