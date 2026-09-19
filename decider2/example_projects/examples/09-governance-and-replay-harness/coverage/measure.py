"""Coverage over rules, tree nodes, gates, table cells, scorecard bins and
reason codes - and the answer to spec §13 Q13 (can it be uniform?).

THE ANSWER IS: TWO PRIMITIVES, NOT ONE AND NOT FIVE.

decider2 collapses most of the question before it is asked. Doc 03 §8.4: there
is ONE type - Module - and three combinators, and branches are not a separate
node kind. Doc 04 §4.1: `branch_path` is a compile-time immediate stored on
whichever branch executes, effectively free at any batch size. So:

  PRIMITIVE 1 - BRANCH PATHS. A rule firing is a branch. A tree node taken is a
  branch. A gate evaluating true is a branch. A ruleset arm, a routing index, a
  loop early-exit: all branches. One int64 per branch point per record, free.
  Rules, nodes, gates and arms are therefore ONE measurement, and the fact that
  they are one measurement is evidence that they are one kind - which is what
  Q5 was asking.

  PRIMITIVE 2 - INDEX READS. A table cell read is not a branch, it is an index.
  A scorecard bin is an index. These need `(table_version, flat_index)` per
  read, which is NOT free: flow 03 reads the rate card tens of times per
  application inside its solve, so per-record cell traces at 22 M decisions a
  year are gigabytes of almost never-read evidence.

  So cell coverage is a COUNTER, not a record: a shared histogram over the flat
  index, incremented in the kernel, merged per partition. Per-record cell reads
  are kept only for the §8.3 detail tier (declines, referrals, disputes, the
  0.5% sample) where they are needed for explanation rather than for coverage.

  That asymmetry is exactly why spec §5.6 says table cell coverage is measured
  but not thresholded, and the reason is now a mechanism rather than a
  concession.

FRAME TIER. Coverage over a month of full volume (22 M/12 decisions, ~400 M
tree-node evaluations for flow 04's monthly cycle alone) in under 2 hours is a
group-by over the branch-path columns. No record-tier work at all.
"""

from __future__ import annotations

import polars as pl


class RuleCoverage:
    rule_id: str
    evaluations: int
    firings: int
    firing_rate: float
    firing_rate_when_reachable: float     # spec §5.8: a rule that fires 100% of the time it is
                                          # reached but is reached 0.2% of the time is a different
                                          # animal from one always reached and never firing.
                                          # Only separable because item 14 records EVALUATION.
    days_since_last_firing: int | None


def rule_coverage(flow: str, window_days: int) -> pl.LazyFrame:
    pass  # group-by over branch_path columns joined to the manifest's rule index


def dead(flow: str) -> pl.LazyFrame:
    """Zero firings in 90 days. Flow 01 adds ~30 rules a month and removes none
    without this; 540 additions since go-live and every fraud rule estate says
    a third stop mattering within a quarter."""
    pass


def not_really_rules(flow: str) -> pl.LazyFrame:
    """Firing on more than 40% of traffic. Not a rule - a policy somebody wrote
    as a rule. The output is not a list, it is a RENAMING PROPOSAL: the report
    names the rule, its firing rate, and the artefact it should move into,
    because 'review it at the level a policy is reviewed at' needs a
    destination or nothing happens."""
    pass


def unreachable_nodes(flow: str) -> pl.LazyFrame:
    """Nodes with zero records in 90 days, AND the condition upstream that
    absorbs them - usually a node re-thresholded three versions ago. Across 60
    trees with several hundred nodes each this is not findable by reading, and
    the upstream attribution is what turns a list into an action."""
    pass


def never_read_cells(family: str, window_days: int) -> "CellCoverage":
    """~11% of the 63 360 Flex cells are read in a year. The other 89% are
    either legitimately unused combinations or errors nobody will ever notice.
    This measurement does not say which. It makes the question askable, and it
    makes a new card's changes to never-read regions correctly uninteresting -
    which is what makes diff/semantic.aggregate() readable."""
    pass


def unreachable_reason_codes() -> tuple[int, ...]:
    """Of 380 registered codes, which have never been raised by ANY flow.
    Cross-flow by construction, because the manifest of each flow declares the
    codes it CAN raise (structural) and this measures which it DOES (empirical).
    A code in neither set is dead; a code declared but never raised is a gap in
    testing or in reality, and the two are distinguishable."""
    pass
