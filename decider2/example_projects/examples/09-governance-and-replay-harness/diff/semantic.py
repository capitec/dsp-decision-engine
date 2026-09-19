"""Semantic diff over two manifests. Never textual, ever.

Spec §5.4: "A diff that reports 63 360 changed cells because the export
ordering changed is worse than no diff, because it trains reviewers to approve
without reading."

doc 08 §6.2 offers `decider2.diff(old, new) -> list[Change]` and doc 04 §5.2
says "a config diff is an audit record ... with no extra machinery". That is
true for PARAMS and false for everything else, which is most of what changes.
Four shapes the framework's `diff` does not have an answer for, each of which
the spec names explicitly (§13 Q11: "value diffs are easy; structural diffs are
where reviewers actually need help"):

  A RULE INSERTED MID-PRIORITY. Every rule after it shifts sequence. A
    positional diff reports 31 changes; the semantic diff reports one insertion
    and - the part reviewers need - the set of rules whose EFFECT changes
    because something now runs before them. In a waterfall where a rule reduces
    to a percentage of the current value, that set is not empty and is not
    obvious. It is computable from the manifest with no data: for each rule
    after the insertion, does its value expression reference the ceiling's
    current value?

  A TREE RE-PARENTED. The node identity map (spec 04 §5.4.3) classifies every
    node as carried_forward / changed / added / removed. A node whose condition
    is unchanged keeps its identity even if a node elsewhere moved - so
    re-authoring a tree in the modelling tool does not renumber the world, and
    month-on-month node volumes stay comparable exactly when they should.

  A STAGE REORDERED. Two modules swap in the `|` expression. Lineage is static,
    so the diff can state precisely which outputs can now be affected by which
    inputs that could not be before. That sentence - "risk_grade can now be
    affected by campaign_id, which it could not in the previous version" - is
    the single most useful line a structural diff can produce and it needs no
    data at all.

  A RATE CARD REFRESHED. 63 360 cells. Aggregate or it is unreadable: see
    aggregate() below and the worked figures in
    artefacts/swapset-REL-2027-04-FLX.md.
"""

from __future__ import annotations

from datetime import datetime

from manifest.model import GovernanceManifest


class Change:
    kind: str                # "rule_inserted" | "cell_moved" | "node_rethresholded" | "stage_reordered"
    what: str                # human sentence, generated
    where: str               # stable identity, never a position
    before: object
    after: object
    author: str              # from the authoring system, by name - not "the release"
    authored_at: datetime    # three distinct timestamps, spec §5.4
    approved_at: datetime
    promoted_at: datetime
    approval_ref: str        # resolves to an approver and a forum, not a date-proximity guess
    estimated_impact: "ImpactEstimate | None"


def diff(old: GovernanceManifest, new: GovernanceManifest) -> tuple[Change, ...]:
    """Structure-aware, identity-keyed, order-insensitive where order is not
    semantic and order-sensitive where it is."""
    pass  # match by stable identity, then compare interiors, then compare params, then aggregate


def effect_shift(old: GovernanceManifest, new: GovernanceManifest) -> tuple[str, ...]:
    """Rules whose text did not change but whose EFFECT did, because something
    upstream in the waterfall now runs before them. Static; no population."""
    pass


def lineage_delta(old: GovernanceManifest, new: GovernanceManifest) -> tuple[str, ...]:
    """'output Z can now be affected by input Y, which it could not before.'
    Pure doc 04 §3, and the thing compliance reviewers ask for by instinct."""
    pass


def aggregate(cells: tuple["CellChange", ...]) -> str:
    """Spec §5.4: aggregation is required, not optional.

    Produces the reviewable sentence:
      "1 840 of 63 360 cells changed; all in grades 7-9; mean move
       +0.31 pp; maximum +0.95; no cell moved down; 1 612 of the changed cells
       are in regions production has never read in 24 months; affects an
       estimated 14.2% of Flex Loan volume."

    The never-read clause is this project's addition and it is the one that
    saves the reviewer's time: coverage/measure.py knows which cells production
    actually reads (11% of the Flex card in a year), so changes to never-read
    regions are correctly uninteresting and are counted rather than listed.
    """
    pass
