"""P13 the solve — the largest single budget in the flow (38.0 ms). Spec 5.14.
31 decision points, none of which is "try the midpoint".

Bounded, terminating, deterministic, correct, tie-broken, attributed and
re-checkable (spec 5.14's seven requirements) are properties this file's
tests hold it to (tests/phases/test_p13_solve.py), not properties asserted in
a docstring. `correctness_parameters=("evaluations_per_term",)` on the P13
envelope (phases/__init__.py) marks the 19-per-term ceiling as UNAVAILABLE to
budget negotiation — when P13's 38 ms is squeezed, this number does not move.

The search is a `Partition` over declared breakpoints — every rate-band edge,
the R14 568 fee kink (fees.py), every cap boundary — not a binary search
assuming monotonicity that spec 5.14's worked failure disproves.
"""

from __future__ import annotations

from decider2 import module, param, Loop

def candidate_domain(product_minimum: float, requested_amount: float, amount_cap: float,
                     product_maximum: float) -> list[float]:
    """R250 grid, from the product minimum to min(requested, amount_cap, product max)."""
    pass  # generate the R250-stepped candidate set within the bounds

def declared_breakpoints(product_code: int, decision_date: str) -> list[float]:
    """Every rate-band edge, the fee's R14 568 kink, every cap boundary — read
    from the artefacts that create them, not re-derived. §13's Q on
    non-monotonicity: this makes 'the search is correct' a theorem about
    THESE declarations, not a hope about the search."""
    pass  # union of rate-card band edges, fee kink, and applicable cap boundaries

def partitioned_search(candidate_domain: list[float], declared_breakpoints: list[float],
                       max_affordable_instalment: float,
                       evaluations_per_term: int = param(19, ge=1, le=19)) -> dict:
    """Evaluates each partition's monotone segment once at its boundary,
    binary-searches WITHIN a segment (safe there because monotonicity is a
    theorem on the segment), and returns the true maximum, its binding
    constraint code, and its evaluation trace."""
    pass  # bounded, deterministic search over the partitioned domain, <= evaluations_per_term

def tie_break(partitioned_search: dict) -> float:
    """Prefer the larger amount; among equal amounts the lower total cost;
    among equal costs the shorter term. A declared rule, never evaluation order."""
    pass  # apply the three-level declared tie-break

def offer_or_no_offer(tie_break: float, partitioned_search: dict) -> dict:
    """No proven maximum within the evaluation ceiling -> refer, queue 6,
    reason 2420. An explicit, recorded, monitored outcome, never a silent
    truncation."""
    pass  # {"amount": tie_break, ...} or {"outcome": "refer", "queue": 6, "reason": 2420}

SolvePerTerm = module(candidate_domain, declared_breakpoints, partitioned_search,
                      tie_break, offer_or_no_offer, name="solve")
