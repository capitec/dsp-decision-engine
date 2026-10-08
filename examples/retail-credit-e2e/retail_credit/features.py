"""Features this flow derives itself: the client segment and the evidence tier of the income."""
from __future__ import annotations

from decider import missing_as, step

SEGMENT_NEW_THIN = 1
SEGMENT_NEW_THICK = 2
SEGMENT_EXISTING_TRANSACTIONAL = 3
SEGMENT_EXISTING_CLEAN = 4
SEGMENT_EXISTING_PAST_ARREARS = 5
SEGMENT_NON_RESIDENT = 10
SEGMENT_STAFF = 11
SEGMENT_JOINT = 12

# employment_type_code -> segment: self-employed, pensioner, social grant, informal
_EMPLOYMENT_SEGMENT = {3: 6, 4: 7, 5: 8, 6: 9}

# credit_core's income source (1 employer, 2 payslip, 3 statement, 4 declared, 5 bureau estimate, 0 none) as this
# product's seven-tier ladder (1 best). Tiers 4 and 5 of the ladder (irregular statements, deposit history) are not
# separable from the source code, so 3 stands for every statement.
_EVIDENCE_TIER = {1: 1, 2: 2, 3: 3, 5: 6, 4: 7, 0: 7}


def income_evidence_tier(income_verification_tier: int) -> int:
    return _EVIDENCE_TIER.get(income_verification_tier, 7)


def segment_code(
    bureau_accounts_ever_count: int = missing_as(0), bureau_history_months: float = missing_as(0.0),
    internal_tenure_months: float = missing_as(0.0), internal_credit_holding: bool = missing_as(False),
    worst_arrears_months: float = missing_as(0.0), employment_type_code: float = missing_as(1.0),
    is_non_resident: bool = missing_as(False), is_staff: bool = missing_as(False),
    is_joint_application: bool = missing_as(False),
) -> float:
    """The first segment that fits, in this order: joint, staff, non-resident, employment type, existing client, new.

    The order is this flow's choice: a staff member who is also self-employed is staff. It is a code order, not a table.
    """
    if is_joint_application:
        return SEGMENT_JOINT
    if is_staff:
        return SEGMENT_STAFF
    if is_non_resident:
        return SEGMENT_NON_RESIDENT
    if employment_type_code in _EMPLOYMENT_SEGMENT:
        return _EMPLOYMENT_SEGMENT[int(employment_type_code)]
    if internal_credit_holding:
        return SEGMENT_EXISTING_PAST_ARREARS if worst_arrears_months >= 2.0 else SEGMENT_EXISTING_CLEAN
    if internal_tenure_months >= 4.0:
        return SEGMENT_EXISTING_TRANSACTIONAL
    if bureau_accounts_ever_count >= 3 and bureau_history_months >= 15.0:
        return SEGMENT_NEW_THICK
    return SEGMENT_NEW_THIN


income_evidence_tier_step = step(income_evidence_tier)
segment_code_step = step(segment_code)
