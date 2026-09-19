"""What a flow looks like from the outside.

This is the structurally unusual part of this project. The harness operates on
eight codebases owned by six teams on different release trains. It must not
import any of them. So a flow, seen from here, is not code - it is a
GOVERNANCE MANIFEST: one signed build artefact per flow build, containing the
flow's graph as data.

decider2 makes this possible and nothing else does. Doc 02 §2 makes structure a
pydantic instance; doc 08 §7.2 requires `from_config(pipeline.to_dict()) ==
pipeline` to round-trip in CI. Those two together mean the manifest is LOSSLESS:
everything the harness reasons about statically - lineage, diffs, renderings,
shadowed logic, the input inventory, the contract - is computable from the
manifest alone, with no flow code in the harness process, ever.

Two planes, and the separation is the whole architecture:

  STATIC PLANE     manifests only. Contract checks, semantic diff, renderings,
                   lineage, shadow analysis, the adjustment register, coverage
                   aggregation, drift, the input inventory. No flow code.
  EXECUTION PLANE  replay, what-if, swap-set, certification. Runs the flow's
                   OWN pinned image in a network-denied sandbox
                   (replay/engines.py). Never the harness's interpretation of
                   the flow.

The manifest is not a description of the flow that might diverge from it.
It IS the flow's graph, serialised - which is the same argument spec 09 §5.7
makes about the reviewable artefact and the deployed logic being "the same
thing rendered twice".
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class CapabilityPin(BaseModel):
    """Spec §5.1 item 5: per capability, not per library. Two majors may be live."""
    capability: str                      # "core.affordability"
    major: int
    version: str                         # "2.4.1"
    contract_hash: str                   # the frozen interface file, doc 03 §5.1


class TableRef(BaseModel):
    family: str                          # "rate_card.flex"
    version: str                         # "RC-FLEX-2027-03"
    content_hash: str
    effective_from: date
    effective_to: date | None
    approver: str
    approval_ref: str
    cells: int                           # 63_360


class LeafInput(BaseModel):
    """Spec §13 Q15. The input inventory is GENERATED, not declared.

    It is `pipeline.schema()`'s unbound-input set - doc 03 §2.2 already
    computes it and already errors on a typo. So "a feature not in the
    inventory" is not possible: the inventory is defined as what the graph
    reads. The problem flips direction, which is the point: what becomes
    detectable is a feature in the governance inventory that NO flow reads
    (dead) and a feature read with no classification (a build error).
    """
    name: str
    dtype: str
    nullable: bool
    null_policy: Literal["required", "missing_as", "optional"]   # doc 03 §1
    source_system: str
    expected_range: tuple[float, float] | list[str] | None
    classification: str                  # explain/disclosure.py lattice
    prohibited_adjacent: bool
    permitted_use_ref: str | None        # required if prohibited_adjacent, spec §5.12


class WitnessField(BaseModel):
    name: str
    qualification: Literal["final", "at_module", "all_versions"]   # doc 03 §7 `term_cap@*`
    classification: str


class GovernanceManifest(BaseModel):
    """One per flow build. ~2-6 MB for flow 01 (610 rules); 180 KB for flow 02."""

    flow: str                            # "flow03.unsecured_granting"
    build: str                           # compiled artefact id, doc 08 §8
    structure_fingerprint: str           # pipeline.fingerprint(), covers skeleton AND interiors
    skeleton_identity: str               # framework + module distribution versions
    declared_variants: dict              # fuse groups, parallel regions, fastmath per kernel
    fallback_set: tuple[str, ...]        # doc 08 §8. MUST BE EMPTY - see contract item 4

    graph: dict                          # pipeline.to_dict(). The lossless part.
    interfaces: dict                     # every module's materialised interface, doc 03 §5.1
    interiors: dict                      # ruleset bodies, tree bodies, table row sets
    params_schema: dict                  # JSON Schema, doc 08 §6.2. Bounds AND owner class.
    params_resolved_digest: str          # what production actually ran on
    lineage_gaps: tuple[str, ...]        # every @breaks_lineage region, by name. Governance, doc 04 §3.

    inputs: tuple[LeafInput, ...]
    witness: tuple[WitnessField, ...]    # derived by attest(), not typed by an author
    tables: tuple[TableRef, ...]
    capabilities: tuple[CapabilityPin, ...]
    reason_codes: tuple[int, ...]        # every code this flow can raise
    claims: tuple["Claim", ...]          # every `holds` sentence, with its last verification

    rendering_ref: str                   # the approved plain-language artefact for THIS build
    approval_ref: str                    # the forum and minute that approved it
    identity_ledger_ref: str             # node identity map vs the previous build (spec §5.15.2)

    def cone(self, changed: tuple[str, ...]) -> frozenset[str]:
        """Outputs reachable from `changed` via static lineage. Drives scoped re-certification."""
        pass  # transitive closure over the graph; no execution, no data
