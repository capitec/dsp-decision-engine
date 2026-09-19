"""Canonicalisation, node identity, edge identity, route identity.

This file is the answer to spec §5.4.3 and to questions 4 and 5 of §13, and it is
the single most opinionated file in the sketch.

THE RULE
--------
    node_key = "n_" + blake2b64(canonical_condition_text ‖ discriminator)

and *nothing else* goes in.  Not ancestry, not depth, not export order, not tree
size, not sibling anything, not the tree id, not the version.

Three consequences, all of them requirements:

  * inserting two levels above a node leaves that node's key alone
    (spec §11.2 — the modelling tool renumbered all fourteen of them);
  * moving a threshold 2200 -> 2600 gives a different key, always
    (spec §5.4.3 req 1);
  * replacing the modelling tool entirely changes nothing, because the key is
    derived from meaning and the new tool exports the same meaning
    (spec §11.12 — two years of node history survives the migration).

THE AWKWARD BIT, MADE EXPLICIT RATHER THAN INFERRED
---------------------------------------------------
Content-only keys collide when one tree tests the same condition in two places
that the analyst considers different tests.  We do not guess.  Default: the same
condition **is** the same node, and the tree is a DAG — which is also the honest
reading of spec §5.3.3, where node 6 is reached from both node 3 and node 4.  One
node, two in-edges; the two arrivals stay distinguishable because the *edge* is
what the route records.

Where the analyst wants two separate nodes, they say so, in the document:

    {"condition": {...}, "as": "risk_gate_late_tenure"}

`as` is a free label carried in the export, hashed into the key, and required by
validation whenever a canonical condition occurs more than once without one.  It
is the mechanism by which spec §13 Q5 ("who decides, and is the decision
reviewable?") has an answer: the analyst decides, in the artefact, and Credit
Risk sees it in the identity map.
"""

from __future__ import annotations

from typing import Literal, NewType, TypedDict

NodeKey = NewType("NodeKey", str)        # "n_" + 16 lowercase hex.  Fixed width, DB-safe.
LeafKey = NewType("LeafKey", str)        # "l_" + 16 lowercase hex.
EdgeId = NewType("EdgeId", int)          # uint64
RouteDigest = NewType("RouteDigest", int)  # uint64
SlotId = NewType("SlotId", str)          # "s:<node_key>:<ordinal>"

CANON_VERSION = 3   # bump = every key in the estate changes.  Never done casually.


# --------------------------------------------------------------------------
# The closed condition algebra.  Total codegen: every kind below has an emitter
# in emit.py and an interval rule in validate.py.  Nothing else may appear.
# --------------------------------------------------------------------------

Op = Literal[
    "ge", "gt", "le", "lt", "eq", "ne",       # numeric comparison vs a slot
    "in_set", "not_in_set",                   # categorical codes, sorted+deduped
    "in_band", "not_in_band",                 # banded feature, resolved to edges
    "is_true", "is_false",
    "is_established", "is_not_established",   # the third null situation, 00 §7.4
    "linear_ge", "linear_le",                 # lhs  op  coef*rhs + const
]


class Atom(TypedDict, total=False):
    op: Op
    feature: str            # feature registry id, never a display name
    rhs_feature: str        # linear_* only
    coef: float             # linear_* only, a slot
    value: float            # comparison, a slot
    codes: list[int]        # in_set, sorted ascending, deduplicated
    band_edges: list[float]  # in_band, RESOLVED at canonicalisation time


def canonicalise(condition: dict, *, band_set_version: str) -> str:
    """Produce the canonical text whose digest is the node key.

    Five normalisations, each one a class of spurious identity break removed:

      1.  the boolean tree is rewritten to OR-of-ANDs; the ANDs are sorted by
          (feature_id, op, literal) and the ORs by their own text, so an analyst
          reordering two conditions in the modelling tool does not renumber
          anything;
      2.  features are named by registry id, so a display-name change is not a
          logic change;
      3.  money literals are scaled int64 cents and rates are scaled int32 basis
          points before hashing, so 3500.0000001 and 3500.0 are one node
          (doc 03 §1.2 — float literals must never reach a hash);
      4.  `in_set` code lists are sorted and deduplicated;
      5.  **band references are resolved to their boundary values** at
          `band_set_version`.  `income_band_code in {5,6,7,8,9}` canonicalises to
          the interval the bands actually denote.  This is what makes a band-table
          edit a genuine identity event for exactly the nodes it moves, and a
          non-event for the rest (spec §6.1 — "a band edit is the same event as a
          threshold edit and must be reviewed as one").  It is also why
          `feature_bands` is an `identity_bearing` Table.  See DEMANDS #22.
    """
    pass  # normalise -> DNF -> sort -> scale literals -> resolve bands -> join


def node_key(condition: dict, *, band_set_version: str, discriminator: str = "") -> NodeKey:
    """blake2b(digest_size=8) over CANON_VERSION ‖ canonical text ‖ discriminator."""
    pass  # "n_" + blake2b(...).hexdigest()


def leaf_key(outcome_code: int, tier_code: int, reason_label: int,
             discriminator: str = "") -> LeafKey:
    """A leaf's identity is the **decision it represents**, not its payload.

    Deliberately excluded: the amount rule, the channel preference list and the
    priority weight.  Those are tunables a campaign owner moves weekly, and
    "response by leaf month on month" (spec §5.4.2(1)) must survive a priority
    weight being nudged from 0.58 to 0.61.  Deliberately included: outcome, tier
    and reason label, because a leaf that moves from tier B to tier C is a
    different offer and its history should visibly break.

    The collision case is the same as for nodes and takes the same remedy: two
    leaves in one tree with identical (outcome, tier, reason) require an `as`.
    """
    pass  # "l_" + blake2b(...)


def edge_id(from_key: NodeKey, direction: bool) -> EdgeId:
    """uint64 identity of one out-edge.  Derived from the node key, so it carries
    the same stability properties for free."""
    pass  # blake2b(from_key ‖ b"T"/b"F", digest_size=8) -> int


def fold_route(edges: list[EdgeId]) -> RouteDigest:
    """The route digest: an order-sensitive 64-bit fold over the edges taken.

        r = 0xcbf29ce484222325
        for e in edges:  r = ((r ^ e) * 0x100000001b3) & MASK64

    Two integer ops per node visited, computed **inside the kernel** as the
    traversal happens.  At 400 M evaluations x 7.4 nodes that is ~6 billion
    integer ops per cycle — call it three seconds — against a tree evaluation
    that is already register-resident.  This is what makes spec §5.4's "the path
    is an output, not a by-product" affordable enough that nobody drops it when
    the run gets tight.

    Because the edges are derived from node keys, **a route that is genuinely the
    same walk has the same digest in March and in April**, across tree versions,
    for free.  That is route-level month-on-month comparability with no extra
    machinery, and it breaks exactly when a node on the route changes.

    Collisions: validation enumerates every feasible route at publication and
    asserts the digests are distinct within the tree version.  On a collision it
    refuses to publish rather than perturbing a salt, because a salt would make
    the digest depend on something other than the walk.
    """
    pass  # FNV-1a over 64-bit edge ids


def slot_id(key: NodeKey, ordinal: int) -> SlotId:
    """`s:<node_key>:<ordinal>` — the overlay-facing name of one tunable literal.

    The node key is INSIDE the slot id, and that is the whole trick behind spec
    §5.9.8 and §11.18.  A volume dial names `s:n_a1f45e9c2b70d863:0`.  If the
    analyst re-fits and node 6's condition moves, its key moves, the slot id
    ceases to exist, and publication of the new tree version **fails** with
    "overlay OV-2026-114 targets a slot this version does not contain".  The dial
    cannot silently reattach to a different node, and it cannot be silently
    dropped either.  Nobody had to write a rule to get that; it falls out of
    putting identity in the name.

    `ordinal` is the position in the *canonical* (sorted) condition, so it does
    not move when the analyst reorders the conditions in the modelling tool.
    """
    pass  # f"s:{key}:{ordinal}"


def shape_fingerprint(document: dict) -> str:
    """The **compile** key.  Covers topology, feature column slots, operator kinds,
    slot-array length and leaf-table shape.  Excludes every literal value.

    So `v11 -> v12` (node 6's threshold 2200 -> 2600) is:
        shape_fingerprint   UNCHANGED   -> artefact cache hit, zero compilation
        node_key(node 6)    CHANGED     -> history break, identity map says so

    One number moving has two different identity consequences depending on which
    artefact it moved in, which is exactly what spec §5.4.3 req 1 and req 6
    demand of each other.  A single `pipeline.fingerprint()` (doc 08 §8) cannot
    express it.  See FRAMEWORK-DEMANDS #2.
    """
    pass  # blake2b over the shape projection of the document
