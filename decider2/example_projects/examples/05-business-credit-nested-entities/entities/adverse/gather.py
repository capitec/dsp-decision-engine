"""Stage 5.6a -- Event -> Entity. THE central file of this sketch.

`Gather` is the ascent. It turns a child collection into parent-grain scalars,
and it is the one construct doc 03 has no equivalent of.

--------------------------------------------------------------------------
WHAT A GATHER IS
--------------------------------------------------------------------------
    Gather(<child grain>, into=<parent grain>, <name> = <fold>, ...)

Each keyword is one parent-grain output name. Each value is a fold over the
child rows of that parent. That is all. The folds are deliberately stupid --
count, sum, min, max, any, all, best_of, collect_of -- and every piece of
cleverness lives in an ordinary child-grain step that the fold references by
name. `weighted_log_odds` in people/blend.py is a step; the fold is `sum_of`.

Three consequences, and they are the reasons this shape was chosen over the
obvious alternative of "write a function that takes the list":

--------------------------------------------------------------------------
1. ORDERING INDEPENDENCE IS STRUCTURAL, NOT TESTED
--------------------------------------------------------------------------
Spec s8 requires that the outcome and the attribution be invariant under
shuffling of the entity list and each entity's event list, demonstrated over
1 000 applications x 20 shuffles. Spec s13 Q7 asks how that is *guaranteed*
rather than hoped for.

`Gather` admits only commutative-associative combiners. **There is no `first`.**
Where the spec wants "the worst event", you write `best_of(..., tie_break=...)`
with a total order on a stable identity, and the framework rejects a `best_of`
whose tie-break is not total. So shuffle-invariance is a build-time property of
every fold in the project, and the randomised test becomes a regression check
rather than the guarantee.

The alternative -- a function receiving a list -- can write `events[0]`, and
`decider2` would have no way to know.

--------------------------------------------------------------------------
2. ATTRIBUTION IS FREE, AND IT IS THE RIGHT SHAPE
--------------------------------------------------------------------------
Every fold carries a **witness**: the set of child rows that contributed to it.

    minor_recent_count = count(where="is_minor_recent")

produces two values at the Entity grain: `minor_recent_count` (int64) and
`witness("minor_recent_count")`. The witness is not something the author writes
and cannot be something the author forgets.

This answers spec s13 Q6 -- "is the attribution shape uniform, or does it vary
by rule kind?" -- with: **uniform**. A count attributes to every row where the
predicate held. A sum attributes to every row it summed. A `best_of` attributes
to one row. A rule that fires on a gathered quantity attributes to that
quantity's witness. s5.6's three requirements fall out:

    "Counts create severity that no individual event has"   -> witness of a count
    "Aggregates create severity that no individual event has" -> witness of a sum
    "Trend and recency modify what counts would give"       -> two witnesses,
                                                               unioned by the rule

--------------------------------------------------------------------------
3. THE REPRESENTATION, AND ITS LIMIT, STATED HONESTLY
--------------------------------------------------------------------------
A witness is a bitset over the child's ordinal within its parent. `Event`
declares `capacity=60`, so a witness is **one int64** and costs one column.
`Entity` declares `capacity=40` -- also one int64.

That is convenient and it is also a coupling worth naming: a governance bound
(40 entities, because that is what the Bank will look through) has landed on a
machine word (64 bits). Above capacity 64 the bitset is not available and the
witness degrades to *re-filtering the child frame by the predicate name*, which
is exact but costs a scan. `Candidate` (capacity 2 200) is in that regime and
needs no witness, so nothing in this project pays it -- but the framework must
implement the fallback correctly rather than merely offering it, because the
next project will have a 200-element collection. FRAMEWORK-DEMANDS D05.

Ordinals never escape the kernel. At materialisation the framework resolves the
bitset against `Event.identity` and the persisted record carries `event_id`
values. Spec s13 Q5 -- "positional identity fails the moment an event is
removed" -- is satisfied because position is an implementation detail with a
lifetime of one kernel invocation.
"""

from decider2 import Gather, count, sum_of, max_of, min_of, any_of, best_of, asc, desc
from grains import Event, Entity


EntityAdverseFacts = Gather(
    Event,
    into=Entity,
    name="entity_adverse_facts",

    # -- counts. AE-R-01, AE-R-02, AE-R-03, AE-R-04, AE-R-05, AE-R-09, AE-R-11
    disqualifying_count  = count(where="is_disqualifying"),
    minor_recent_count   = count(where="is_minor_recent"),
    minor_total_count    = count(where="is_minor"),
    material_24m_count   = count(where="is_material_within_24m"),
    material_recent_count= count(where="is_material_within_recency"),
    dishonoured_12m_count= count(where="is_dishonoured_within_12m"),
    immaterial_count     = count(where="counts_toward_clear"),
    event_count          = count(),

    # -- aggregates. AE-R-06 and AE-R-07. Note `sum_of` over an Optional column:
    #    a null unsatisfied amount is excluded from the sum AND recorded in
    #    `unsatisfied_unknown_count`, because s4.4 is explicit that a judgment
    #    with no recorded amount is not a judgment for R0. A fold that silently
    #    coerced null to zero would understate the aggregate, which is the
    #    direction that approves business it should not.
    unsatisfied_total    = sum_of("unsatisfied_amount", where="is_unsatisfied",
                                  nulls="exclude_and_count"),

    # -- trend. AE-R-10's two terms, gathered separately and differenced at the
    #    Entity grain by an ordinary step, because a fold that computed a
    #    difference would have to carry two witnesses and the framework's fold
    #    vocabulary stays at one witness each.
    trailing_12m_count   = count(where="in_trailing_window"),
    preceding_12m_count  = count(where="in_preceding_window"),

    # -- exceptions. AE-R-12's carve-out for AE-C-18 / AE-C-20.
    has_write_off_or_fraud = any_of("is_write_off_or_fraud"),

    # -- summary for the committee pack (s9.3 item 3) and for the per-entity
    #    summary in s7.1. `best_of` lifts the winning child's fields to the
    #    parent, which is how "worst event" and "most recent event" become
    #    parent-grain values without a second query.
    worst_event = best_of(
        "event_severity_code",
        tie_break=(desc("unsatisfied_amount"), asc("event_id")),
        lift=["event_id", "event_type_code", "amount", "event_date",
              "classifying_rule_id"],
    ),
    most_recent_event = best_of(
        "event_date",
        tie_break=(asc("event_id"),),
        lift=["event_id", "event_type_code", "event_date"],
    ),

    # -- provisional. AE-C-22's consequence has to reach the Entity verdict:
    #    a verdict any of whose contributing events is disputed is itself
    #    provisional.
    any_provisional = any_of("classification_provisional"),

    # -- overlay sensitivity. s5.6: a verdict must carry
    #    `verdict_overlay_sensitive` where any contributing event was classified
    #    against an overlaid threshold. This is a fold over a value the overlay
    #    machinery generated at the Event grain, so the author writes one line
    #    and nothing about it can drift from the overlay that caused it.
    any_overlaid_threshold = any_of("material_threshold_was_overlaid"),
    overlay_ids_seen       = any_of.bitor("threshold_overlay_id_mask"),
)


# --------------------------------------------------------------------------
# The same Event collection, rolled up a SECOND way, for a different consumer:
# the per-entity adverse summary in s7.1 and s9.3. Different folds, different
# predicates, same children. This is what "three consumers roll the same
# elements up differently" looks like -- three Gathers, not one Gather with
# three consumers arguing about its output.
# --------------------------------------------------------------------------
EntityAdverseSummary = Gather(
    Event,
    into=Entity,
    name="entity_adverse_summary",
    severity_immaterial_count = count(where="counts_toward_clear"),
    severity_minor_count      = count(where="is_minor"),
    severity_material_count   = count(where="is_material"),
    severity_disq_count       = count(where="is_disqualifying"),
    synthetic_count           = count(where="is_synthetic"),
    aggregate_all_amounts     = sum_of("amount", nulls="exclude_and_count"),
    oldest_event_age          = max_of("event_age_months"),
    newest_event_age          = min_of("event_age_months"),
)
