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

# credit_core's income source (1 employer, 2 payslip, 3 statement, 4 declared, 5 bureau estimate, 0 none) as this
# product's seven-tier ladder (1 best). The ladder's tiers 4 and 5 (irregular statements, deposit history) cannot be told
# apart from the source code, so 3 stands for every statement.
def income_evidence_tier(income_source_code: int) -> int:
    if income_source_code in (1, 2, 3):
        return income_source_code
    return 6 if income_source_code == 5 else 7


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
    if 3 <= employment_type_code <= 6:  # self-employed, pensioner, social grant, informal: segments 6 to 9
        return employment_type_code + 3
    if internal_credit_holding:
        return SEGMENT_EXISTING_PAST_ARREARS if worst_arrears_months >= 2.0 else SEGMENT_EXISTING_CLEAN
    if internal_tenure_months >= 4.0:
        return SEGMENT_EXISTING_TRANSACTIONAL
    if bureau_accounts_ever_count >= 3 and bureau_history_months >= 15.0:
        return SEGMENT_NEW_THICK
    return SEGMENT_NEW_THIN


income_evidence_tier_step = step(income_evidence_tier)
segment_code_step = step(segment_code)
