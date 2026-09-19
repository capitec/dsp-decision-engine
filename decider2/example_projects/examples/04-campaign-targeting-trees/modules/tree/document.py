"""The published tree artefact, as validated data.

Two documents exist per tree version and the distinction matters:

    trees/v11.export.raw.json   what the modelling tool emitted.  Retained for
                                provenance.  Never read by the engine.
    trees/v11.json              the canonical published artefact.  Derived.

Spec §13 Q7 asks whether the published artefact is the export or something
derived, and if derived whether the derivation is reviewable by the analyst who
authored it.  **Derived**, and reviewable: `decider tree canonicalise` emits the
canonical document plus `v11.review.md`, a side-by-side of the tool's node
numbering against the assigned node keys with the canonical condition text next
to the exported one.  The analyst signs the canonical form, not the export.  The
export's sha256 is carried in the canonical document so the pair is joined
forever.

Note what is *in* the document that is strictly derived: `node_key`, `slot_index`
and `shape_fingerprint`.  They are denormalised into the artefact so that the
warehouse and the offline renderer can read them without running any code
(spec §5.4.1(b)).  Validation recomputes and asserts equality, so the
denormalisation cannot drift from the logic that ran (spec §13 Q6).
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from .canonical import LeafKey, NodeKey, SlotId


class SourceExport(BaseModel):
    tool: str
    tool_version: str
    export_sha256: str
    exported_at: str
    sample_window: tuple[date, date]
    fitted_by: str


class Node(BaseModel):
    node_key: NodeKey                      # derived; asserted, then re-derived
    condition: dict                        # the closed algebra of canonical.py
    discriminator: str = ""                # the `as` label, "" for almost all nodes
    on_true: NodeKey | LeafKey
    on_false: NodeKey | LeafKey
    slots: dict[SlotId, float]             # base values of this node's literals
    author_label: str = ""                 # free text from the modelling tool, e.g. "L3 risk gate"


class LeafPayload(BaseModel):
    leaf_key: LeafKey
    leaf_outcome_code: int = Field(ge=0, le=2)     # 0 do-not-target, 1 target, 2 target-as-control
    offer_tier_code: int | None = Field(None, ge=1, le=7)
    reason_label: int = Field(ge=1, le=9999)       # registries/reason_labels.csv
    # The amount rule is a closed 3-kind algebra, not an expression string
    # (doc 08 §3.2).  See DEMANDS #14 for why "register a step per coefficient"
    # is not a usable answer here.
    amount_rule: Literal["none", "fixed", "capped_preassessment", "scaled_capped"]
    amount_cap_cents: int | None = None
    amount_scale: float | None = None
    channel_pref: list[int] = []                   # ordered channel codes
    priority_weight: float = Field(0.0, ge=0.0, le=1.0)


class TreeDocument(BaseModel):
    artefact: Literal["campaign_tree"]
    campaign_id: int
    tree_id: int
    tree_version: int
    effective_from: date
    effective_to: date | None
    variant_label: str = "champion"        # champion / challenger_a / challenger_b

    # Everything identity depends on, pinned in the artefact so that a replay in
    # 2029 resolves the same meaning (00 §7.3, spec §5.1 req 2).
    canon_version: int
    feature_registry_version: str
    band_set_version: str

    root: NodeKey
    nodes: list[Node]
    leaves: list[LeafPayload]

    # Derived, asserted, recomputed at validation.
    shape_fingerprint: str
    slot_index: dict[SlotId, int]          # slot id -> position in the thresholds array
    reads: list[str]                       # 9..74 feature ids; drives column extraction
    source_export: SourceExport
