"""P16 -- Offer assembly and cross-product arbitration (spec 10 §5.17): product 10 only.

Full-width P16 compares offers across products and campaigns (10 §5.17);
this project's slice runs product 10 alone, so arbitration collapses to
"pick the loop's result if L1 fired, otherwise the initial solve's
result" -- still a real decision (which of two live candidates the record
reflects), just not a multi-product one. `is_recommended` is always the
one offer this slice can produce; the spec's 0..8 offer array (10 §7.1)
is a declared simplification to exactly one, consistent with SCOPE.md's
"Skip products 11-40... in depth".
"""
from __future__ import annotations

from decider import missing_as, step


def assemble_offer(
    is_eligible: bool,
    fraud_verdict_code: int,
    product_10_eligible: bool,
    affordability_verdict_code: int,
    solved_amount: float,
    solved_instalment: float,
    solve_binding_constraint: str,
    solve_rate_card_cell_id: str,
    solved_nominal_annual_rate: float,
    loop_pass_index: int = missing_as(0),
    loop_solved_amount: float = missing_as(0.0),
    loop_solved_instalment: float = missing_as(0.0),
    loop_binding_constraint: str = missing_as(""),
) -> tuple[float, float, str, bool]:
    """(offer_amount, offer_instalment, offer_binding_constraint, has_offer). If L1 fired
    (`loop_pass_index > 0`) and produced a solved amount, its result supersedes the initial
    pre-loop solve -- it is a re-derivation against hypothetical obligations, not an
    alternative to compare against the first (10 §5.23.1's diagram has no "keep the better
    of the two"; the loop exists because the first one failed).
    """
    if not is_eligible or fraud_verdict_code == 3 or not product_10_eligible:
        return 0.0, 0.0, "", False
    if loop_pass_index > 0 and loop_solved_amount > 0.0:
        return loop_solved_amount, loop_solved_instalment, loop_binding_constraint, True
    if affordability_verdict_code in (1, 2) and solved_amount > 0.0:  # PASS, MARGINAL
        return solved_amount, solved_instalment, solve_binding_constraint, True
    return 0.0, 0.0, "", False


assemble_offer_step = step(assemble_offer, outputs=("offer_amount", "offer_instalment",
                                                      "offer_binding_constraint", "has_offer"))
