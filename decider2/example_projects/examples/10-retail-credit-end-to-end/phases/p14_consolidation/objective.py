"""P14(e) objective and the anti-harm rule — 8 decision points. Spec 5.15(e).

The objective is CONFIGURATION (five named objectives, blendable by weight,
per channel, effective-dated — a `params` document, not a structural change),
changed by Credit Risk Policy without a release. The anti-harm rule is the
one hard-coded gate: no amount of objective tuning may bypass it.
"""

from __future__ import annotations

from decider2 import module, param

def objective_score(scenario_result: dict, objective_weights: dict = param(
        {"instalment_relief": 0.4, "total_cost": 0.2, "blended_rate": 0.1,
         "bank_expected_value": 0.2, "client_outcome_score": 0.1})) -> float:
    pass  # weighted blend of the five named objectives, per the resolved weights

def anti_harm_verdict(scenario_result: dict, baseline_total_cost: float,
                      longest_settled_remaining_term_months: int,
                      total_cost_delta_threshold: float = param(0.35, ge=0, le=1),
                      term_extension_threshold_months: int = param(48, ge=0)) -> bool:
    """Rejects where total_cost_delta > 35% of baseline, or the term extension
    exceeds 48 months beyond the longest settled account's remaining term —
    unless a documented client instruction overrides it, recorded with the
    consultant's identifier."""
    pass  # apply both thresholds; True means the scenario is REJECTED

def rank(objective_score: float, anti_harm_verdict: bool) -> dict:
    """Ranks surviving (non-rejected) scenarios by objective_score. Feeds the
    loop's promotion logic (loops/l1_consolidation.py) and P16's arbitration."""
    pass  # rank scenarios where anti_harm_verdict is False (not rejected)

Rank = module(objective_score, anti_harm_verdict, rank, name="objective")
