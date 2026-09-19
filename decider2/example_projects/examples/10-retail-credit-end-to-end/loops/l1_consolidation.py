"""L1 — the four-phase feedback loop. Spec 5.23, 5.23.1. Affordability fails,
consolidation searches, affordability re-runs against a hypothetical
settlement, pricing and the solve re-run, offer assembly runs, and — if
nothing shippable resulted and the bound is not reached — it happens again.

THIS IS THE FILE FRAMEWORK-DEMANDS #10 IS ABOUT: doc 03 §8.3's `Loop` combinator
is a single body module iterating on its own carried values. This loop's body
is FOUR PHASES — P10, P14, P12, P13, P16 — each independently owned, budgeted
and tested, and none of it may be a copy of them. `ConsolidationLoop` is
therefore a `Loop` whose body is an EXPRESSION over the same P10/P12/P13/P14/P16
objects flow.py composes elsewhere in the graph, not four new modules that
happen to look like them. There is exactly one P10 in this codebase; this file
calls it, at a different site tag (`P10.at_site("loop")`, values/bases.py's
site vocabulary), inside a loop, under a hypothetical ambient basis.

Bound: 4 passes (`max_iterations=4`), not a timeout — a time-based bound is
not reproducible, and `loop_termination_code` must be one of five REASONS, not
a clock reading. `should_continue` is the termination guarantee itself: a
pass runs only if it can strictly improve (the candidate space shrinks, or the
objective improves by >= R25 of instalment relief). A pass that can do neither
does not run, and that is a property of the INPUT, not of elapsed time.
"""

from __future__ import annotations

from decider2 import Loop
from decider2.basis import Hypothetical, ambient

from phases import P10, P12, P13, P16
from phases.p14_consolidation import objective, scenarios, settleability

P14_BODY = settleability.Classify | scenarios.Generate | scenarios.Evaluate | objective.Rank


def should_continue(best_objective_so_far: float, candidates_remaining: int,
                    loop_idx: int, min_improvement: float = 25_00) -> bool:
    """The termination guarantee, spec 5.23.1: strictly improve or stop. Not a
    convergence heuristic — a comparison against the declared minimum."""
    pass  # candidates_remaining > 0 and (loop_idx == 0 or improved_by >= min_improvement)


def loop_body(actual_or_prior_hypothetical_obligations: float, loop_idx: int):
    """One pass. Runs P14's search under a HYPOTHETICAL basis this function
    sets (never typed inside P10/P12/P13 themselves — values/bases.py's
    ambient rule), then re-invokes affordability, pricing, the solve and offer
    assembly at the loop's own site tag."""
    pass  # with ambient(Hypothetical(scenario_ref=...)): P14_BODY | P10.at_site("loop") \
         #     | P12 | P13 | P16, tagging every emitted value loop_pass_index=loop_idx


ConsolidationLoop = Loop(
    should_continue,
    loop_body,
    carries=["best_offer", "best_objective_so_far", "candidates_remaining",
             "discarded_offers"],
    max_iterations=4,
    # Entry condition: L1 is entered only when P10's actual-basis verdict is 3
    # (fail) AND consolidation_eligible (set once, in P01 — see
    # phases/p01_admission/routing.py's `consolidation_eligible` step). This
    # is what makes P14 CONDITIONAL on entry point 1 (spec 4.6) without a
    # second flow: the condition lives here, once, not as an `if` duplicated
    # at every call site that might reach it.
    entry_condition="p10.affordability_verdict_code == 3 and consolidation_eligible",
    # The nine things the record must show about a loop that ran four times
    # (spec 5.23.1) fall out of `carries` plus `loop_pass_index`: every
    # `discarded_offers` entry retains its own trigger, its own basis, and the
    # phase that rejected it (P16's anti-harm or P17's assertion), because
    # those are ordinary emitted values tagged with THIS loop's pass index —
    # nothing extra was invented to keep three rejected offers on file.
).named("l1_consolidation")

# Budget: 180 ms of the 820 ms interactive-loop class (budgets/ep01_loop.toml)
# is L1's own overhead; the repeated P10/P12/P13 passes take the remaining
# ~460 ms (spec 5.24.1). L4 (the small circular solve inside
# phases/p14_consolidation/scenarios.py's required_advance) is nested INSIDE
# this loop's body, not a sibling of it — two loops, one visibly inside the
# other, exactly as spec 5.23's loop table shows it.
