"""Stage 9 — how a tree gets to production.  This does not run in the cycle.

15 to 40 tree changes a month, of which roughly 30 are threshold moves and 5 are
structural.  A tree must reach production **without an engineer** (spec §5.9).

The two paths, and the fact that they cost differently is the design working:

    THRESHOLD MOVE (30/month)          STRUCTURAL CHANGE (5/month)
    -------------------------          ---------------------------
    new tree_version document          new tree_version document
    shape_fingerprint UNCHANGED        shape_fingerprint CHANGED
    artefact cache HIT                 artefact cache MISS -> compile, ~24 s for
    zero compilation                     a 400-node tree, measured and reported
    node keys for touched nodes move   node keys for touched nodes move
    identity map: n changed            identity map: carried/added/removed/changed
    published in ~90 seconds           published in ~3 minutes

Both paths run every validation.  Neither path lets an engineer near it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from modules.tree import (build_identity_map, canonicalise, enumerate_routes,
                          validate_tree)
from modules.tree.emit import emit_portable_interpreter, emit_tree_kernel
from modules.tree.routes import publish_node_meta
from modules.overlays.resolve import resolve as resolve_overlays


def publish(export_path: Path, *, campaign_id: int, effective_from: date,
            snapshot, submitted_by: str) -> dict:
    """The whole path from a modelling-tool export to a live tree version.

    1.  canonicalise      export -> canonical document + v<n>.review.md
                          The analyst signs the canonical form.  The export's
                          sha256 is carried in the document forever.
    2.  validate          the eight checks of §5.9 plus the two implied ones.
                          Rejection names the node and the problem
                          (acceptance criterion 2).
    3.  identity map      against the previous version, with last cycle's node
                          volumes attached.
    4.  overlay check     every overlay in force must still be applicable
                          (§5.9.8).  A dangling slot target rejects here.
    5.  routes            enumerate, digest, assert distinct, publish.
    6.  emit + compile    IF the shape fingerprint is new.  Compilation happens
                          HERE, in this process, weeks before the cycle that runs
                          it, and the result lands in the content-addressed
                          artefact cache.  `prepare()` at cycle time is a lookup.
                          DEMANDS #6, #7.
    7.  node_meta         warehouse rows for the version.
    8.  portable          the dependency-free interpreter ships with the artefact.
    9.  sign-off          campaign owner + Credit Risk for any credit campaign,
                          both signing ONE artefact bundle:
                             the tree, the identity map, the population impact
                             with and without overlays, the disposition of every
                             overlay in force, the validation report.
   10.  manifest          a manifest amendment naming the new version and its
                          effective_from.  The cycle resolves versions from
                          manifests, never from a directory listing.
    """
    pass


def impact(campaign_id: int, candidate, *, as_at: date, snapshot,
           overlays: str = "both") -> dict:
    """Population impact for the publication gate.

    This is doc 08 §5's `decider2.impact(active, candidate, sample)` with three
    things it does not have and needs here (DEMANDS #15):

      * `as_at=` — resolve the comparison's artefacts by the cycle date, not by
        today, so a back-dated publication is evaluated correctly;
      * `overlays="both"` — the with/without pair §5.9.8 requires;
      * the comparison is between two *versions of one data-shaped module*, over
        a 14.2 M snapshot, not between two pipeline generations over a sample.
        60 campaigns x 40 publications a month is 2 400 impact runs; each one is
        two tree kernels over one campaign's candidate population.
    """
    pass


def band_edit_impact(feature_id: str, new_bands) -> dict:
    """Spec §6.1, and the question §13 Q12 asks: *how are shared band definitions
    owned when 38 trees depend on them and a band edit is functionally a
    simultaneous edit to all 38?*

    Answer: it **is** a simultaneous edit to all 38, and the system says so.
    Because `canonicalise` resolves band references to their boundary values, a
    band edit changes the node key of exactly those nodes whose tested band edges
    moved — across every tree, at once.  This function reports them:

        feat:income_band_code  bands 5..9 lower edge 12 000 -> 12 800
        affected: 38 trees, 114 nodes, est. 412 000 clients crossing
          campaign 23  node n_7d38c2a05f6941be -> n_1d94e0b5ac762f38  ~214 000
          campaign 31  node n_0a5e93c17bd4f286 -> ...                 ~61 000
          ...

    and publication of the band set is blocked until each affected campaign owner
    has an identity map for their tree.  "A band edit that quietly moves 400 000
    clients across node 5 of campaign 23 is the same event as a threshold edit
    and must be reviewed as one" — so it produces the same artefact.
    """
    pass


def trees_referencing(feature_id: str, *, as_at: date) -> list[dict]:
    """Scenario 4 (a feature is retired from the mart) and scenario 5 (Compliance
    rules a feature a proxy).  Eleven trees reference it; the impact must be known
    **before** the retirement, not after the cycle fails.

    Static, no execution, no cycle.  A reverse index from feature id to tree
    version, maintained at publication.  Doc 04 §3's `lineage()` runs forward from
    an output inside one pipeline; this runs backward from an input across 60
    independently versioned artefacts, which is a different query the framework
    does not currently have.  DEMANDS #23.
    """
    pass
