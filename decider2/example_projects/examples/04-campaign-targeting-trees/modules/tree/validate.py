"""Publication validation — spec §5.9, all eight checks, plus the two the spec
implies but does not number.

A submitted tree is a *submission* until this passes.  Nothing downstream can
observe a tree that has not passed, because publication is what writes the
artefact into the version register, and the cycle resolves versions from the
register only.  A tree reaches production without an engineer; it does not reach
production without this file.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from .document import TreeDocument


class Finding(BaseModel):
    check: str
    severity: str            # "reject" | "raise" | "note"
    node_key: str | None
    message: str


class ValidationReport(BaseModel):
    campaign_id: int
    tree_version: int
    submitted_at: str
    canon_version: int
    findings: list[Finding]
    population_impact: dict          # with and without the overlay stack, §5.9.7+8
    feature_list: list[str]          # the regulator pack's raw material, §9.2(1)
    route_count: int
    emitted_lines: int
    compile_seconds: float           # MEASURED at publication, never estimated
    shape_fingerprint: str
    verdict: str                     # "published" | "rejected"


def validate_tree(doc: TreeDocument, *, as_at: date, snapshot, overlays) -> ValidationReport:
    """Run every check.  Returns a report; `verdict == "rejected"` on any finding
    with severity "reject"."""
    pass  # dispatch to the checks below, assemble the report, emit the artefacts


# --- §5.9.1 ---------------------------------------------------------------
def check_total(doc: TreeDocument) -> list[Finding]:
    """Every path terminates in a leaf; every node has both out-edges bound; no
    edge points outside the document; the graph is acyclic."""
    pass


# --- §5.9.2 ---------------------------------------------------------------
def check_reachability(doc: TreeDocument) -> list[Finding]:
    """No unreachable node: no node whose conditions cannot be satisfied given the
    conditions on **every** path leading to it.

    This is not graph reachability.  It is satisfiability of a conjunction of
    atoms from the closed algebra, and it is why the algebra is closed:
    `ge/gt/le/lt/eq/ne` over a feature reduce to an interval, `in_set` to a code
    set, `in_band` to a union of intervals, `linear_*` to a half-plane over two
    features.  Every operator has a rule; an operator without one cannot be added
    (DEMANDS #14).

    In the September review 1 340 of 9 200 live nodes had received no traffic in
    90 days.  This check separates the ones that are structurally dead — a defect
    — from the ones that are empirically dead, which is a signal to the campaign
    owner and not a rejection.
    """
    pass  # interval/set propagation down every path; UNSAT at a node -> reject


# --- §5.9.3 ---------------------------------------------------------------
def check_no_contradiction(doc: TreeDocument) -> list[Finding]:
    """`age >= 65` beneath `age < 60`.  Same machinery as check_reachability,
    reported separately because the remedy differs: a contradiction is an
    authoring mistake, an unreachable node is often a stale branch."""
    pass


# --- §5.9.4 ---------------------------------------------------------------
def check_features(doc: TreeDocument, registry) -> list[Finding]:
    """Every referenced feature exists in the mart, with a compatible type, and —
    where banded — a band definition covering the values tested.

    Also the reverse direction, which the spec makes a requirement in §4.1 and
    scenario 4: the feature registry is **frozen per mart version** and the tree
    pins `feature_registry_version`.  Data Engineering widening a feature from
    int8 to int16 produces a new registry version, which fails every tree pinned
    to the old one **before the cycle**, not after it.  Preventing, not detecting.
    """
    pass


# --- §5.9.5 ---------------------------------------------------------------
def check_leaf_outcomes(doc: TreeDocument, registries) -> list[Finding]:
    """Known tier, permitted channel for this campaign, reason label in the
    registry, priority weight in 0..1, and an amount rule that cannot exceed the
    `core.appetite` maximum for any grade the campaign can reach.

    "Cannot exceed" is checked symbolically against the appetite grid, not
    sampled: `scaled_capped(scale, cap)` over a pre-assessed amount already
    bounded by appetite is safe iff `scale <= 1.0`.  `amount_scale > 1.0` is a
    reject, which is spec §5.5 req 1 turned into a publication gate rather than a
    monitoring metric.
    """
    pass


# --- §5.9.6 ---------------------------------------------------------------
def check_prohibited(doc: TreeDocument, prohibited_register) -> list[Finding]:
    """No feature on the prohibited register appears, directly or as a declared
    proxy.  Proxy declarations are transitive and the register grows: scenario 5
    is a postcode-derived affluence index ruled a proxy with 30 days' notice,
    against 11 trees.  `impact.trees_referencing("feat:postcode_affluence_idx")`
    answers "which 11" statically, before the ruling takes effect (DEMANDS #23).
    """
    pass


# --- §5.9.7 and §5.9.8 ----------------------------------------------------
def check_population_impact(doc: TreeDocument, snapshot, overlays, bound: float) -> list[Finding]:
    """Estimated population impact against the most recent snapshot, within the
    campaign's declared bound (default +/-25%), reported **twice**: with the
    overlay stack in force and with it disabled.

    The two-number requirement is the whole point of §5.9.8.  A tree that is only
    within bound because a volume dial is holding it there doubles overnight
    three weeks later when the dial expires, and the with/without pair is what
    makes that visible at sign-off instead of at 06:00 on the first of the month.
    """
    pass


def check_overlay_disposition(doc: TreeDocument, overlays) -> list[Finding]:
    """§5.9.8 proper.  For every overlay in force over this campaign, the
    submitted version must still admit it:

      * a threshold shift names `s:<node_key>:<ordinal>`.  If the node's condition
        moved, the key moved, the slot is gone -> **reject**, naming the overlay,
        the old node key and the candidate successor from the identity map.
      * a cap reduction scoped to a tier the new version no longer produces ->
        **reject**.
      * an overlay whose scope matches nothing in the new version -> **reject**
        (spec §5.3.4 req 6: a tightening scoped to a campaign retired last month).

    Never "silently reinterpreted against a different node".  This is the check
    that scenario 18 exists to exercise, and it costs nothing because the slot id
    already carries the node key.
    """
    pass


# --- implied, not numbered ------------------------------------------------
def check_identity_collisions(doc: TreeDocument) -> list[Finding]:
    """Two nodes with the same canonical condition and no `as` discriminator; two
    leaves with the same (outcome, tier, reason) and no `as`.  Reject with the
    two author labels so the analyst can say which is which — *"node keys collide
    for 'L3 risk gate' and 'L5 risk gate re-test'; if these are the same test say
    nothing, if they are different tests give one an `as` label"*."""
    pass


def check_route_budget(doc: TreeDocument, max_routes: int = 50_000) -> list[Finding]:
    """Enumerable feasible routes within budget, and every route digest distinct.

    The budget is a real constraint on the analyst and it is stated rather than
    discovered: a 12-level tree that fans out without merging can exceed it.  Over
    budget, the route dictionary stops being a dictionary and the path artefact
    falls back to an explicit edge list per evaluation — still inside the 120 GB
    bound at ~60 bytes per evaluation, but it loses the cheap node-volume join.
    We would rather refuse and make the analyst merge branches.
    """
    pass
