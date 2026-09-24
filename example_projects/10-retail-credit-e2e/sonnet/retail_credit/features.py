"""P06 -- Feature derivation (spec 10 §5.7): the part this project owns.

Income, deductions, expense norms and obligations are `core.income`,
`core.deductions`, `core.expense_norms` and `core.obligations` --
project 00's units, wired directly into `pipeline.py` exactly as project
00's own demo pipeline wires them (see `credit_core` NOTES.md "What I
publish"; this is DEPS.md's "hidden coupling", 10 §5.7(b): those three
units must live in 00, or this project stops being standalone, and they
do). This module holds only what 10 itself owns: **segment assignment**
(10 §4.3's twelve segments -- the defining condition, not the library's
concern) and the handful of derived risk/behavioural features P09's cap
register and P07's scorecard read directly.
"""
from __future__ import annotations

from decider import missing_as, step

SEGMENT_NEW_THIN = 1
SEGMENT_NEW_THICK = 2
SEGMENT_EXISTING_TRANSACTIONAL = 3
SEGMENT_EXISTING_CLEAN = 4
SEGMENT_EXISTING_PAST_ARREARS = 5
SEGMENT_SELF_EMPLOYED = 6
SEGMENT_PENSIONER = 7
SEGMENT_SOCIAL_GRANT = 8
SEGMENT_INFORMAL = 9
SEGMENT_NON_RESIDENT = 10
SEGMENT_STAFF = 11
SEGMENT_JOINT = 12

_EMPLOYMENT_SEGMENT = {3: SEGMENT_SELF_EMPLOYED, 4: SEGMENT_PENSIONER, 5: SEGMENT_SOCIAL_GRANT, 6: SEGMENT_INFORMAL}

# `core.income.income_verification_tier` is a *source* code (1 employer-confirmed .. 5
# bureau-estimated, 0 none), enumerated for identification, not ordered by evidence
# severity. 10 §5.7(b)'s seven-tier evidence ladder (employer-confirmed best, declared-only
# worst) is this project's own policy reading of that source -- 00 supplies the capability
# (which source won), 10 supplies the calibration (how severely each source is treated).
# Statement-derived-regular vs. irregular and internal-deposit-history (spec tiers 3-5) are
# not distinguished by 00's unit at this depth, so they collapse to tier 3 here -- a
# declared simplification, not a silent one.
_EVIDENCE_TIER_SEVERITY = {1: 1, 2: 2, 3: 3, 5: 6, 4: 7, 0: 7}  # income_verification_tier -> 10's own tier 1..7


def income_evidence_tier(income_verification_tier: int) -> int:
    return _EVIDENCE_TIER_SEVERITY.get(income_verification_tier, 7)


income_evidence_tier_step = step(income_evidence_tier)


def assign_segment(
    bureau_accounts_ever_count: int = missing_as(0),
    bureau_history_months: float = missing_as(0.0),
    internal_tenure_months: float = missing_as(0.0),
    internal_credit_holding: bool = missing_as(False),
    worst_arrears_months: float = missing_as(0.0),
    employment_type_code: int = missing_as(1),
    is_non_resident: bool = missing_as(False),
    is_staff: bool = missing_as(False),
    is_joint_application: bool = missing_as(False),
) -> int:
    """One condition per segment, evaluated in the spec's own declared precedence (10 §4.3
    lists them 1..12; this project treats an earlier row as higher precedence where a client
    could match more than one, e.g. staff who are also self-employed).
    """
    if is_joint_application:
        return SEGMENT_JOINT
    if is_staff:
        return SEGMENT_STAFF
    if is_non_resident:
        return SEGMENT_NON_RESIDENT
    if employment_type_code in _EMPLOYMENT_SEGMENT:
        return _EMPLOYMENT_SEGMENT[employment_type_code]
    if not internal_credit_holding and internal_tenure_months >= 4.0:
        return SEGMENT_EXISTING_TRANSACTIONAL
    if internal_credit_holding:
        return SEGMENT_EXISTING_PAST_ARREARS if worst_arrears_months >= 2.0 else SEGMENT_EXISTING_CLEAN
    if bureau_accounts_ever_count >= 3 and bureau_history_months >= 15.0:
        return SEGMENT_NEW_THICK
    return SEGMENT_NEW_THIN


assign_segment_step = step(assign_segment, output="segment_code")
