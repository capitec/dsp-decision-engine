"""`decider govern export` - what a flow team runs, once, in its own CI.

This is the whole onboarding cost of a ninth flow, and it is deliberately one
command with no arguments the team has to think about:

    uv run decider govern export pipelines/unsecured.py \
        --params config/unsecured/production.json \
        --inventory inventory/flow03.yaml \
        --out dist/manifest/flow03-<build>.json

It exits non-zero if `contract.check` grades the flow C. So a flow that cannot
be governed cannot produce the artefact its own release pipeline requires,
which is how the contract becomes a build property rather than a memo.

DEVIATION FROM DOC 07. Doc 07 §5 gives `decider export --params` and
`--interiors`, split by document kind "because they version independently".
Correct, and insufficient here: neither carries the capability pins, the
fallback set, the declared variants, the claims or the input classifications,
and spec 09 §5.1 needs all five to replay. `govern export` is a third
document, and it is a UNION rather than a third independently-versioned thing -
it is keyed on the build, and it is immutable.
"""

from __future__ import annotations

from pathlib import Path

from decider2 import Pipeline

from contract.check import Attestation, check
from manifest.model import GovernanceManifest


def export(pipeline: Pipeline, *, params_doc: dict, inventory: Path,
           origin: str, build: str) -> tuple[GovernanceManifest, Attestation]:
    """Produce the manifest and its attestation from a built pipeline."""
    pass  # walk the graph, materialise interfaces, resolve params complete=True, pin capabilities, check


def verify_roundtrip(m: GovernanceManifest, pipeline: Pipeline) -> None:
    """`from_config(m.graph) == pipeline`, asserted here as well as in the framework's CI.

    Doc 08 §7.2 wants this to stop the data model drifting. This project wants
    it for a harder reason: if it ever fails, every static answer the harness
    has given about that flow - lineage, diff, shadowing, the rendering the
    committee approved - was about a different graph from the one that ran.
    """
    pass  # raises ManifestDrift, which is an incident, not a warning


def publish(m: GovernanceManifest, a: Attestation) -> str:
    """Write-once, content-addressed, with an independently verifiable digest (spec §9.3)."""
    pass  # returns the immutable address; the harness can read it and can never rewrite it
