"""Stage 7 -- verdict, and the three shapes of answer (spec 02 §5.7).

00's `core.affordability.affordability_verdict_code` is a general two-number
comparison, correctly scoped to the unit (DEPS.md "Cycles" §2: 00 owns the
units' arithmetic and interface). What is 02's alone is layered here: a
named `indeterminate` distinct from `fail` (§5.7.1), and the fact that the
*shape* of the answer -- (a) pass/fail against a known instalment, or
(b) capacity with no instalment supplied -- depends on nothing but whether
`proposed_instalment` was passed, never a second code path (§5.7.2).
"""
from __future__ import annotations

from decider import param

PASS, MARGINAL, FAIL, INDETERMINATE = 1, 2, 3, 4

# evidence_sufficiency_code (02 §4.3: "why an assessment is indeterminate; zero when not").
EVIDENCE_OK = 0
EVIDENCE_APPLICANT1_INCOME_UNESTABLISHED = 1
EVIDENCE_APPLICANT2_INCOME_UNESTABLISHED = 2
EVIDENCE_BOTH_APPLICANTS_INCOME_UNESTABLISHED = 3
EVIDENCE_BUREAU_STALE = 4
EVIDENCE_REFER_ACCOUNT = 5
EVIDENCE_INCOME_BELOW_MINIMUM_TIER = 6
EVIDENCE_NO_NET_INCOME = 7

_GAP_TO_EVIDENCE_CODE = {
    1: EVIDENCE_APPLICANT1_INCOME_UNESTABLISHED,
    2: EVIDENCE_APPLICANT2_INCOME_UNESTABLISHED,
    3: EVIDENCE_BOTH_APPLICANTS_INCOME_UNESTABLISHED,
}


def evidence_sufficiency_code(
    income_verification_tier: int, minimum_income_tier: int,
    applicant_income_evidence_gap: int, bureau_is_stale: bool, has_refer_account: bool,
    net_monthly_income: float,
) -> int:
    """§5.7.1: named causes, checked in the order the ombud would ask about them --
    whose evidence is missing, before what the resulting number would have been."""
    if applicant_income_evidence_gap in _GAP_TO_EVIDENCE_CODE:
        return _GAP_TO_EVIDENCE_CODE[applicant_income_evidence_gap]
    if bureau_is_stale:
        return EVIDENCE_BUREAU_STALE
    if has_refer_account:
        return EVIDENCE_REFER_ACCOUNT
    if income_verification_tier == 0 or income_verification_tier > minimum_income_tier:
        return EVIDENCE_INCOME_BELOW_MINIMUM_TIER
    if net_monthly_income is None or net_monthly_income <= 0:
        return EVIDENCE_NO_NET_INCOME
    return EVIDENCE_OK


def affordability_verdict_code(
    evidence_sufficiency_code: int, max_affordable_instalment: float,
    proposed_instalment: float | None = None,
    marginal_band: float = param(0.05, ge=0.0, le=1.0),
) -> int:
    """§5.7.1 + §5.7.2. `indeterminate` is decided first and can never be reached through
    `fail` (the "most consequential error available in this project"). With no
    `proposed_instalment` this is shape (b), capacity: "the verdict is `pass` or
    `indeterminate` only" (§5.7.2(b)) -- there is nothing to fail against.

    Monotone in `proposed_instalment` by construction (§5.7.2(c), 03's requirement): this
    function only ever compares `proposed_instalment` against `max_affordable_instalment`,
    which does not depend on `proposed_instalment` (see `capacity.py`), so a larger proposed
    instalment can only move the comparison towards `fail`, never away from it -- tested in
    `tests/test_monotonicity.py` across band edges, buffer boundaries and the residual floor.
    """
    if evidence_sufficiency_code != EVIDENCE_OK:
        return INDETERMINATE
    if proposed_instalment is None:
        return PASS
    if proposed_instalment <= max_affordable_instalment:
        return PASS
    if proposed_instalment <= max_affordable_instalment * (1.0 + marginal_band):
        return MARGINAL
    return FAIL


def discretionary_income_after(
    discretionary_income: float, proposed_instalment: float | None = None,
) -> float | None:
    """§4.3: "discretionary income once the proposed instalment is committed" -- only
    meaningful for shape (a); `None`, not zero, when no instalment was proposed."""
    if proposed_instalment is None:
        return None
    return round(discretionary_income - proposed_instalment, 2)
