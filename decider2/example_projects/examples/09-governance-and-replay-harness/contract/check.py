"""Checking compliance rather than asserting it.

`check()` never imports a flow. It reads a GovernanceManifest, which is a build
artefact the flow's own CI produced (manifest/export.py). Two consequences:

  - a ninth flow is onboarded by producing a manifest, with NO CHANGE TO THIS
    CODE (spec 09 acceptance criterion 16);
  - six release trains never link into one process, which is the thing that
    would make this harness a bottleneck and therefore a thing teams route
    around.

The harness is in scope for its own regime (spec 09 §9.2), so `check` is run
over the harness's own manifest too, and `harness/tolerances.py` is a versioned
effective-dated artefact like any rate card.
"""

from __future__ import annotations

from datetime import date

from contract.items import ITEMS, GRADES, Proof
from manifest.model import GovernanceManifest


class Finding:
    item: int
    verdict: "Literal['pass', 'fail', 'stale', 'unknown']"
    evidence: str            # a sentence a non-engineer can read
    locus: str | None        # "modules/limit_engine/book_read.py:41" - where to go and fix it
    disables: tuple[str, ...]


class Attestation:
    flow: str
    build: str               # compiled artefact id, doc 08 §8
    structure_fingerprint: str
    as_at: date
    grade: GRADES
    findings: tuple[Finding, ...]
    signature: str           # over the manifest digest; the harness cannot alter evidence (spec §9.3)


def check(manifest: GovernanceManifest, *, corpus: "Corpus | None" = None) -> Attestation:
    """Run all 23 items. STRUCTURAL items need no corpus; EMPIRICAL ones do."""
    pass  # dispatch each Item.check over the manifest, grade, sign, store


def grade_of(findings: tuple[Finding, ...]) -> GRADES:
    """C if any STRUCTURAL fails; B if only EMPIRICAL/DECLARED are stale; else A."""
    pass  # the grade is what appears on every artefact the flow produces


# ---------------------------------------------------------------------------
# Spec 09 §13 Q18 - what happens when a flow does not comply?
# ---------------------------------------------------------------------------
#
# The honest institutional answer, and it is not "block it": the harness has no
# power to stop a live flow deciding, and a governance function that claims that
# power and does not have it is worse than one that does not claim it.
#
# What the harness does have is the certification gate. So:
#
#   grade C  ->  the flow keeps running; it CANNOT OBTAIN a certification
#                artefact (release/certify.py refuses to emit one); the release
#                approver has nothing to sign; and - the part with teeth -
#                every explanation, every pack and every monthly report
#                generated from that flow's decisions carries the banner
#                produced by `degradation_notice()` below.
#
# That last is the mechanism. A grade-C flow's decline letter says, in
# Compliance's words, that the Bank cannot fully reconstruct how the decision
# was reached. No team survives that for two quarters.

def degradation_notice(a: Attestation, audience: "Audience") -> str:
    """The sentence that appears on every artefact derived from a non-compliant flow."""
    pass  # e.g. "Evidence for this decision is incomplete: account state as at the decision date was not retained."


def disabled_capabilities(a: Attestation) -> frozenset[str]:
    """Union of `disables` over failing items. Printed at the top of the attestation."""
    pass  # {"exact_replay", "what_if", "historical_swap_set"} for a flow failing item 8
