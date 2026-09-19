"""The 90-day backtest. 420 M events, 90 minutes, either overlay mode.

The equivalence requirement (spec §5.17, acceptance criterion 6) is the reason
this file is six lines of composition and not a second implementation:

    Interdiction = Normalise | Enrichment | Decision
    Backtest     = ReadRecords          | Decision

`Decision` is the same object. It is not "the same logic", it is the same
pydantic instance, compiled from the same source, reading the same feature
vector columns. There is no path by which the two could disagree, because there
is only one of them.

Two things the spec says make backtesting hard, and where each lands:

1. "The backtest must use RECORDED feature values, never recomputed ones."
   Structural here: `Decision`'s leaf inputs are exactly the feature vector, and
   `ReadRecords` supplies them from the decision record store. `Decision` has no
   ability to recompute a velocity aggregate because it has no ability to call
   anything — every step in it is pure and every input is a column. The failure
   mode the spec is warning about is not prevented by discipline; it is
   unreachable.

2. "Production ran against however many rule set versions and overlay stacks
   were live across those 90 days."
   That is `PartitionByEffective`, below.
"""

from __future__ import annotations

import polars as pl

from decider2 import pipeline
from decider2.frame import Aggregate, Filter, Join, PartitionByEffective, bitset_explode

from fraud_interdiction.decision.applicability import Applicability
from fraud_interdiction.decision.hard_blocks import HardBlocks
from fraud_interdiction.decision.overlays import OverlayStack
from fraud_interdiction.decision.resolve import OverlayAttribution, Resolve, ResolveBase
from fraud_interdiction.ruleset import LiveRules, ShadowRules

# Exactly the tail of pipelines/interdict.py, extracted so both use one object.
Decision = (
    HardBlocks | Applicability | OverlayStack
    | LiveRules | OverlayAttribution | Resolve | ResolveBase | ShadowRules
)


def backtest(
    records: pl.LazyFrame,
    *,
    baseline: str = "as_at_event",     # | "as_at_today" | "as_at:2026-09-01T00:00:00Z"
    overlay_mode: str = "as_it_was",   # | "disabled" | "as_proposed"
    candidate=None,
):
    """Run `Decision` over 90 days of recorded feature vectors.

    `PartitionByEffective` is the invention that makes this honest without
    breaking doc 02 §4's "params do not vary by record". Thresholds ARE
    effective-dated per entry (config/interdict/production.params.json), so over
    90 days an event's applicable threshold depends on its own timestamp. Rather
    than making params per-record — which would destroy the fixed-type guarantee
    that makes retuning free — the frame tier partitions the 420 M events by
    which generation was in force and invokes the kernel once per partition:

        ~300 threshold generations over 90 days
        -> ~300 invocations of ~1.4 M events each
        -> params stay per-invocation, results are per-record correct

    This is a frame-tier operation and it belongs there: it is a group-by over a
    temporal join, which is exactly what polars is for and exactly what the
    record tier must not try to do. FRAMEWORK-DEMANDS.md #13.

    `baseline="as_at_today"` collapses it to one partition, which is the other
    admissible answer and must be stated in the result — "a result that does not
    state which mode produced it is not admissible at the Committee".
    """
    pass


def projected_columns(candidate) -> list[str]:
    """Only the feature-vector columns the candidate rule set can read.

    420 M events x 380 columns is ~1.7 TB and is the actual binding constraint
    on the 90-minute budget — not the kernel, which is ~14 core-minutes of
    compiled scalar work. A candidate rule set touching 12 features reads 12
    columns.

    The projection comes from `Decision.lineage(...)`, computed without
    executing anything. So static lineage — argued for in doc 02 §5 as a
    governance property — turns out to be this project's principal performance
    mechanism. That is worth saying out loud, because it is the strongest
    argument in the doc set for declarative frame operations and nobody made it.
    """
    pass  # union of lineage("live_fire_bits") over the candidate's rules


def metrics(scored: pl.LazyFrame, outcomes: pl.LazyFrame) -> pl.LazyFrame:
    """The eight committee metrics (spec §5.17), all frame-tier.

      hit rate, precision, incremental catch, false positive rate,
      overlap matrix, value blocked, operational load by queue, action churn

    `incremental catch` — "confirmed fraud caught by the proposal that no
    currently live rule caught" — is a bitset AND-NOT and a count, which is why
    the firing set being a fixed-width bitset rather than a variable-length list
    matters for analysis and not only for storage.
    """
    pass


def rule_level_aggregates(scored: pl.LazyFrame) -> pl.LazyFrame:
    """Per-minute firing counts by rule and by overlay.

    `bitset_explode` turns 10 uint64 columns into (rule_index, fired) pairs
    against the generation's rule catalogue. Without it this is 635 hand-written
    shift-and-mask expressions, which is the kind of thing that gets written
    once and then silently stops matching the catalogue. FRAMEWORK-DEMANDS.md #7.

    These aggregates are what make rule-level circuit breakers, overlay impact
    monitoring, dead-rule detection and retirement evidence possible WITHOUT
    scanning 140 M records a month.
    """
    pass  # bitset_explode(...) | Aggregate(by=["minute", "rule_index"], ...)


def daily_equivalence_check(production_day: pl.LazyFrame) -> pl.LazyFrame:
    """Acceptance criterion 6, run every day as a job.

    Replay a sample of yesterday's events through `Decision` and assert equality
    of action_code, fired_rule_ids in order, fired_on_overlay_ids,
    shadow_fired_rule_ids and the reason set. Any mismatch is an incident.

    This is `assert_modes_agree` from doc 03 §11 with a fourth rung added:
    `score() ≡ apply()`. Doc 02 §3.1's ladder covers interpreted/stepped/fused
    within one entry point and says nothing about the two entry points agreeing
    with each other — and they take different fusion decisions by default
    (score fuses maximally, apply never fuses). FRAMEWORK-DEMANDS.md #22.
    """
    pass
