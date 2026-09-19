"""The evaluation budget.

Two bounds that do two different jobs, and the whole point of this file is that
they are not interchangeable.

    candidates   decides the ANSWER.      Must be reproducible.
    wall_ms      decides whether we DEGRADE.  Cannot be reproducible, so it is
                 never allowed to decide the answer directly.

Spec 5.5.3 says a time-based cut-off is not reproducible and demands a
deterministic candidate-count bound as well. That is necessary and not
sufficient: if the wall clock can stop the search at candidate 287, the answer
still depends on how loaded the box was, and acceptance criterion 2 ("replayed
six months later... reproduces the winner, the runners-up, the rejection set and
the pricing to the cent") fails on a busy Tuesday.

So the wall clock may only choose between DECLARED RUNGS of a ladder. Evaluation
proceeds tier by tier; the clock is read between tiers and never inside one. A
tier that starts is a tier that finishes. The result is a deterministic function
of (plan, completed_tier_count), and completed_tier_count is recorded. Replay
pins it and gets the same answer on any box.

This costs something real and it is stated in FRAMEWORK-DEMANDS D6: the kernel
runs three times per interactive assessment instead of once, and the per-call
overhead is paid three times. At 400 candidates that overhead is noise; at 40
(batch) it is not, which is why the batch profile declares a single tier.
"""

from decider2 import module, param, step
from decider2.search import Budget, TerminationCause


# --- the budget declaration -------------------------------------------------
#
# Budget is not a params model. It is an object the Search combinator reads at
# plan time, because the tier ladder has to be known before any candidate is
# generated - the plan is truncated to `candidates`, and the tiers partition the
# plan, so a tier boundary that moved after generation would change which
# candidates are in which tier and therefore which survive a degradation.

InteractiveBudget = Budget(
    name="search_budget.interactive",
    candidates=param(400, ge=1, le=2000, description="Deterministic bound. Decides the answer."),
    wall_ms=param(900, ge=50, le=5000, description="Safety bound. Decides only whether we degrade."),
    tiers=param([120, 280, 400], description="Cumulative. Each is a whole frame through one kernel."),
    min_tier=param(120, description="Below this the search completes late rather than shrinking."),
    on_wall_exhausted="stop_at_completed_tier",
)

BatchBudget = Budget(
    name="search_budget.batch_identification",
    candidates=param(40, ge=1, le=200),
    wall_ms=param(60, ge=10, le=500),
    tiers=param([40]),
    min_tier=param(40),
    on_wall_exhausted="stop_at_completed_tier",
)

RestructureBudget = Budget(
    name="search_budget.restructure",
    candidates=param(260, ge=1, le=1000),
    wall_ms=param(900, ge=50, le=5000),
    tiers=param([90, 180, 260]),
    min_tier=param(90),
    on_wall_exhausted="stop_at_completed_tier",
)


# --- what the budget records ------------------------------------------------
#
# Spec 5.5.2: budget exhaustion is a recorded fact, not a silent truncation. A
# client whose search was truncated is in a different position from one whose
# space was exhausted, and the contact centre needs to know which.


@step(output="termination")
def classify_termination(
    candidates_generated: int,
    candidates_evaluated: int,
    tiers_completed: int,
    tiers_declared: int,
    wall_ms_consumed: float,
    params,
) -> TerminationCause:
    """Say why the search stopped: space exhausted, count bound, or a degraded tier.

    SPACE_EXHAUSTED  the plan generated fewer candidates than the bound. The client
                     was offered everything the ordering rules could construct.
    COUNT_BOUND      the plan was truncated at `candidates`. There were more
                     candidates and the top N were taken, by the ordering in force.
    TIER_DEGRADED    the clock stopped us after tier k of n. Reproducible given k.
    LATENCY_BREACH   even min_tier overran. Recorded, not hidden, and it is a
                     latency defect rather than a credit one.
    """
    pass  # four-way classification over the counts above; no clock read here


@step(output="search_evidence")
def assemble_search_evidence(
    termination: TerminationCause,
    ordering_set_id: str,
    ordering_set_version: str,
    plan_digest: str,
    budget_digest: str,
    objective_set_id: str,
    overlay_stack_digest: str,
    table_version_set: dict,
    params,
) -> dict:
    """Pin everything a replay needs: orderings, plan, budget, objective, overlays, tables.

    `plan_digest` is a content hash of the ORDERED candidate list, not of the
    inputs that produced it. That is deliberate: it lets project 09 prove two
    assessments explored the same space without re-running the plan, and it is
    the cheapest possible answer to "was the search the same?".

    `overlay_stack_digest` is non-optional. Spec 5.5.6: a replay that does not
    pin the stack is not a replay, and the difference it produces will be
    reported as a defect.
    """
    pass  # dict assembly; every field is already computed, nothing is derived here


SearchEvidence = module(
    classify_termination,
    assemble_search_evidence,
    name="search_evidence",
    contract="contracts/search_evidence.json",
)
