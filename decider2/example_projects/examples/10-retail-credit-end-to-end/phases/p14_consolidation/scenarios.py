"""P14(b)+(d) scenario generation and evaluation — 17 + 13 = 30 decision
points. Spec 5.15(c),(d). At most 250 evaluated of up to 2 million priced
combinations, via 16 ordered, owned generation rules resolving to a total
order — never a heuristic search with no declared stopping rule.

Budget exhaustion is a RECORDED FACT (`max_live_scenarios=250` on the P14
envelope), not a silent truncation: `loop_termination_code` distinguishes
"ran out of candidates" from "ran out of budget", and a deterministic
candidate-count bound exists alongside the millisecond one because a
time-based cutoff alone is not reproducible.

`Generate` calls `settleability.Classify`'s post_settlement_state; `Evaluate`
re-invokes P06's obligation logic, P10, P12 and P12's inner arithmetic
between 7 and 12 times PER SCENARIO — 1 750 to 3 000 invocations inside
820 ms. Both are declared here as calls, not copies.
"""

from __future__ import annotations

from decider2 import module, param, Collection

def generation_rule_order(client_settlement_need: dict) -> list[str]:
    """16 ordered rules: highest-rate-first, highest-instalment-first, all-
    arrears, all-external, client-nominated, and 11 more. Owned by Credit
    Risk Policy. Resolves to a total order, ties broken on account reference
    then product code then term."""
    pass  # return the declared generation-rule sequence for this assessment

def candidate_settlement_sets(settleable: list[bool], generation_rule_order: list[str],
                              max_scenarios: int = param(250, ge=1, le=250)) -> Collection:
    pass  # enumerate settlement sets per the rule order, capped at max_scenarios

def hypothetical_obligations(candidate_settlement_sets: Collection,
                             post_settlement_state: list[dict]) -> float:
    """Re-invokes obligations.Derive's treatment over the REDUCED inventory.
    The BASIS coordinate this produces is `hypothetical`, `scenario_ref`-keyed
    — values/bases.py — never typed here; the caller's ambient basis carries it."""
    pass  # re-run obligation treatment excluding settled accounts, including new facility

def required_advance(candidate_settlement_sets: Collection,
                     hypothetical_obligations: float,
                     buffer_pct: float = param(0.018, ge=0),
                     buffer_cap: float = param(2_800_00, ge=0)) -> float:
    """A small circular solve: settlements + buffer + new money + capitalised
    fees + product-specific costs. Declared bound of 5 iterations, R1
    tolerance — L4 in loops/l1_consolidation.py's docstring table."""
    pass  # contraction iteration, <= 5 passes, tolerance R1

def scenario_result(required_advance: float, hypothetical_obligations: float) -> dict:
    """Invokes P10 (Chain), P12 (Rate|Fees|Premium|Instalment) and P13's inner
    arithmetic under the ambient hypothetical basis, 7-12 times."""
    pass  # price and measure this scenario end to end

def budget_exhaustion_fact(candidate_settlement_sets: Collection) -> dict:
    """RECORDED, never silent: {'consumed': n, 'exhausted_by': 'candidates'|'budget'}."""
    pass  # report consumption against max_scenarios and against the ms budget

Generate = module(generation_rule_order, candidate_settlement_sets, name="generate")
Evaluate = module(hypothetical_obligations, required_advance, scenario_result,
                  budget_exhaustion_fact, name="evaluate")
