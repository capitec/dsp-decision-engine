"""`decision_tree` — the module kind campaign targeting is built on.

A `decision_tree` is a **data-shaped module** in doc 08 §3's sense: its interface
is declared in Python (below), its body arrives as a validated document authored
outside engineering, and a UI/analyst edit can never change what `lineage()`
returns.  It is *not* a `ruleset`.  A ruleset is a flat list of independent
`when`/`then` rules; a targeting tree is a rooted DAG whose interior nodes have
two out-edges, whose successors are shared, and whose traversal is the product.

Three things this kind adds that doc 08 §3 does not have, and that the rest of
this directory implements:

  1.  **Node identity** (`canonical.py`) — content-derived, ancestry-free, stable
      under insertion, unstable under any change of meaning.
  2.  **Route capture** (`routes.py`) — a `uint64` folded in-kernel over the
      edges taken, plus a published route dictionary that explodes it in the
      warehouse.  One recording, two grains (spec §5.4 Q3).
  3.  **Two fingerprints** — `shape_fingerprint` is the compile key, `node_key`
      is the identity key.  Threshold moves change the second and not the first,
      which is why 30 of the 40 monthly tree changes cost zero compilation.

Every numeric literal in a published tree is a **slot** in a `Table`, never an
immediate in the emitted source, so a slot value may move without recompiling and
an overlay may substitute one at run time without touching the artefact
(spec §5.3.4 req 1, §5.4.3 req 6).
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import polars as pl

from decider2 import Table, data_shaped_kind

from .canonical import NodeKey, RouteDigest, SlotId, canonicalise
from .document import TreeDocument, LeafPayload
from .emit import emit_tree_kernel
from .identity_map import IdentityMap, build_identity_map
from .routes import RouteDictionary, enumerate_routes
from .validate import ValidationReport, validate_tree

__all__ = [
    "CampaignTree", "TreeDocument", "LeafPayload", "NodeKey", "RouteDigest",
    "SlotId", "IdentityMap", "RouteDictionary", "ValidationReport",
    "canonicalise", "build_identity_map", "enumerate_routes", "validate_tree",
]


# --------------------------------------------------------------------------
# The kind declaration.  This is code, reviewed in git, and changes ~yearly.
# --------------------------------------------------------------------------

CampaignTree = data_shaped_kind(
    name="campaign_tree",

    # `reads` is an UPPER BOUND, not a description (doc 08 §3 consequence note).
    # The actual column set is computed per tree version from the document and is
    # what the boundary extracts — 9..74 columns, never 600.
    reads=["feature:*"],                       # closed by the feature registry
    reads_bound_source="registries/feature_registry.json",

    writes=[
        "leaf_key",                 # NodeKey of the leaf reached
        "leaf_outcome_code",        # int8   target / do-not-target / target-as-control
        "offer_tier_code",          # int8
        "advertised_amount_cents",  # int64  money is scaled int64 (doc 03 §1.2)
        "channel_pref_code",        # int32  packed ordered channel list
        "priority_weight",          # float64
        "reason_label",             # int16
        "route_digest",             # uint64 THE path (spec §5.4)
        "route_depth",              # uint8  nodes visited, for cheap sanity checks
    ],

    # Tunables that are *not* in the document: overlay-substitutable slots live in
    # `thresholds`; leaf payload values live in `leaf_values`.  Both are Tables
    # (arrays), not pydantic fields — see FRAMEWORK-DEMANDS #4.
    tables={
        "thresholds": Table(dtype=pl.Float64, length="shape.slot_count",
                            identity_bearing=True),
        "leaf_values": Table(dtype=pl.Int64, length="shape.leaf_slot_count",
                             identity_bearing=False),
    },

    # Compilation strategy.  NOT the generic-kernel path of doc 08 §3.4: a
    # 400-node DAG dispatched through an interpreter pays an indirect branch per
    # condition and forfeits short-circuiting, which doc 02 §8.2 measures at 7.8x
    # at depth 50.  Codegen, but per *shape*, cached by `shape_fingerprint`.
    strategy="codegen",
    emit=emit_tree_kernel,
    compile_key="shape_fingerprint",
    max_emitted_lines=6_000,        # NOT doc 02 §1.2's 500 — see DEMANDS #8

    # Publication-time obligations.  A document that fails any of these is not a
    # tree version; it is a rejected submission (spec §5.9).
    validate=validate_tree,
    publishes=[
        "node_meta",        # warehouse table, per (campaign, tree_version)
        "route_dictionary", # warehouse table, per (campaign, tree_version)
        "identity_map",     # vs the previous version, retained indefinitely
        "validation_report",
        "portable_interpreter",   # the offline renderer's evaluator (DEMANDS #12)
    ],

    # Re-run partition.  Every module in the cycle declares one; `Arbitration`
    # declares `cycle` and that is what makes spec Q10 answerable statically
    # rather than at 03:00.  See pipelines/monthly_cycle.py.
    partition="campaign",
)


def load_tree(campaign_id: int, as_at: date, *, variant: int = 0) -> TreeDocument:
    """Resolve the tree version in force for `campaign_id` at `as_at`, for `variant`.

    Never "the latest" (spec §5.9).  Resolution is by effective dating over the
    campaign's published versions and by the variant design register.
    """
    pass  # select the version whose [effective_from, effective_to) contains as_at


def prepare(campaign_ids: list[int], as_at: date) -> dict[int, "CompiledTree"]:
    """Load 60 compiled trees from the content-addressed artefact cache.

    Preparation is a cache **lookup**, never a compile (spec §8, preparation cost
    < 4 minutes and independent of population size).  Compilation happened at
    publication, possibly weeks earlier, in another process.  A miss raises; it
    does not compile.  See FRAMEWORK-DEMANDS #6 and #7.
    """
    pass  # resolve versions -> shape_fingerprints -> artefact cache -> raise on miss
