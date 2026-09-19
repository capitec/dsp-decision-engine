"""P10 evidence modes — selects evidence rules, param sets and which outputs
are produced. Spec 5.11. Never selects different ARITHMETIC: the build
checks it (`modes_share_arithmetic=True`, phases/__init__.py) by requiring all
four modes to share chain.py's step set and differ only in `param()` bindings
and `missing_as` policy. A mode that needs a different step does not compile.
"""

from __future__ import annotations

from decider2 import module, param

def assessment_mode_code(entry_point_code: int, loop_pass_index: int) -> int:
    """1 origination, 2 limit, 3 campaign, 4 scenario (inside P14)."""
    pass  # derive from entry_point_code and whether this call is inside the loop

def income_evidence_policy(assessment_mode_code: int) -> dict:
    """Tiers 1-4 origination; tier 5 dominant on limit; tier 5 only on
    campaign; inherited from origination inside a scenario re-evaluation."""
    pass  # select the permitted evidence tier set for this mode

def obligations_policy(assessment_mode_code: int) -> str:
    """"actual" on origination/limit/campaign; "hypothetical" inside P14's
    scenario re-evaluation — selecting the BASIS coordinate the ambient
    invocation already set (values/bases.py), not typing a new one."""
    pass  # return which of existing_obligations' permitted bases this mode reads

def buffer_policy_adjustment(assessment_mode_code: int,
                             campaign_buffer_uplift_pp: float = param(0.03, ge=0, le=0.10)
                             ) -> float:
    """Campaign mode: standard grid + 3pp. Everything else: 0."""
    pass  # campaign_buffer_uplift_pp if assessment_mode_code == 3 else 0.0

SelectEvidenceMode = module(assessment_mode_code, income_evidence_policy,
                            obligations_policy, buffer_policy_adjustment, name="modes")
