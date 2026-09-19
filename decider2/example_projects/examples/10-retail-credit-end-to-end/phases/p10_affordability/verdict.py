"""P10 verdict — four values, four consequences. Spec 5.11.

`degradation="REFUSED"` on the envelope (phases/__init__.py): there is no
degraded affordability mode, deliberately. What exists instead is degraded
EVIDENCE (income_evidence_policy in modes.py falling to a worse tier), which
routes through `verdict 4 indeterminate`, never through a guessed answer.
"""

from __future__ import annotations

from decider2 import module, param

def affordability_verdict_code(max_affordable_instalment: float,
                                minimum_viable_instalment: float = param(185.0, ge=0),
                                evidence_tier: int = 1) -> int:
    """1 pass, 2 marginal (within 8% of minimum, or tier 6-7), 3 fail
    (below minimum — routes to loops/l1_consolidation.py if eligible),
    4 indeterminate (evidence insufficient for the product's minimum tier)."""
    pass  # compare max_affordable_instalment against minimum_viable_instalment and tier

def marginal_conditions(affordability_verdict_code: int) -> dict:
    """Verdict 2 forces outcome_code 'approve with conditions' and restricts
    terms to <= 36 — a consequence read by P16, not decided here."""
    pass  # {"restrict_terms_to": 36} if verdict == 2 else {}

Verdict = module(affordability_verdict_code, marginal_conditions, name="verdict")
