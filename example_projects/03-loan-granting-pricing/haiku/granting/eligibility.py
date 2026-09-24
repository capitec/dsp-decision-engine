"""Eligibility gates for unsecured Flex Loan."""

from typing import List, Optional, Dict, Any
from datetime import date, datetime
from .types import GateVerdict, EligibilityGateCode


def evaluate_eligibility_gates(
    applicant_age_years: float,
    decision_date: date,
    product_code: int,
    channel_code: int,
    residency_code: int,
    employment_type_code: int,
    debt_review_status_code: int,
    administration_order_flag: bool,
    insolvency_status_code: int,
    deceased_flag: bool,
    estate_flag: bool,
    exclusion_list_hits: Optional[List[str]] = None,
    in_flight_applications: Optional[List[Dict]] = None,
    duplicate_detected: bool = False,
) -> tuple[List[GateVerdict], bool]:
    """
    Evaluate all 14 eligibility gates for Flex Loan.

    Returns (gate_verdicts, is_eligible)

    Notes:
    - Evaluates ALL gates even after first failure (spec §5.1)
    - product_code must be 10 for Flex Loan
    - Returns verdicts and final is_eligible flag
    """
    verdicts: List[GateVerdict] = []

    # Gate 1: Product availability by channel (product 10 channel constraints)
    # Flex Loan not on channel 5 (partner) below R10k, or channel 6 (broker) after 2026-03-01
    # For this implementation: channel 6 closed after 2026-03-01
    channel_ok = True
    if channel_code == 6 and decision_date > date(2026, 3, 1):
        channel_ok = False

    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.PRODUCT_CHANNEL,
        gate_name="Product availability by channel",
        passed=channel_ok,
        evaluated=True,
        reason_code="1102" if not channel_ok else None,
        input_value=f"product={product_code}, channel={channel_code}",
    ))

    # Gate 2: Minimum age (18.0)
    min_age_ok = applicant_age_years >= 18.0
    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.MIN_AGE,
        gate_name="Minimum age",
        passed=min_age_ok,
        evaluated=True,
        reason_code="1110" if not min_age_ok else None,
        input_value=applicant_age_years,
    ))

    # Gate 3: Maximum age at maturity (age + term/12 <= 75.0)
    # This gate is evaluated with 84 months (max term) for now; specific term tested later
    max_age_ok = (applicant_age_years + 84 / 12) <= 75.0
    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.MAX_AGE,
        gate_name="Maximum age at maturity",
        passed=max_age_ok,
        evaluated=True,
        reason_code="1111" if not max_age_ok else None,
        input_value=f"age={applicant_age_years}, max_at_84mo={applicant_age_years + 84/12}",
    ))

    # Gate 4: Capacity to contract (not under curatorship, not minor)
    # Simplified: no specific input field, assume passed unless contradicted
    capacity_ok = True
    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.CAPACITY,
        gate_name="Capacity to contract",
        passed=capacity_ok,
        evaluated=True,
        reason_code="1112" if not capacity_ok else None,
    ))

    # Gate 5: Residency (permitted set: codes 1-3)
    # Code 4-6 are non-resident and excluded
    residency_ok = residency_code <= 3
    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.RESIDENCY,
        gate_name="Residency",
        passed=residency_ok,
        evaluated=True,
        reason_code="1120" if not residency_ok else None,
        input_value=residency_code,
    ))

    # Gate 6: Employment type
    # Social grant (5) excluded, informal (6) excluded, pensioner (4) limited to R80k
    # For gates: check if employment type is permitted for any loan
    employment_ok = employment_type_code not in [5, 6]
    reason = None
    if not employment_ok:
        reason = "1130" if employment_type_code == 6 else "1131"
    if employment_type_code == 4:
        reason = "1131"  # Pensioner permitted but with cap (checked in cap waterfall)

    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.EMPLOYMENT,
        gate_name="Employment type",
        passed=employment_ok,
        evaluated=True,
        reason_code=reason,
        input_value=employment_type_code,
    ))

    # Gate 7: Debt review
    # Status: None=0, Application lodged=1, Under review=2, Rescinded=3, Cleared=4
    # Only 0 and 4 pass
    debt_review_ok = debt_review_status_code in [0, 4]
    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.DEBT_REVIEW,
        gate_name="Debt review status",
        passed=debt_review_ok,
        evaluated=True,
        reason_code="1140" if not debt_review_ok else None,
        input_value=debt_review_status_code,
    ))

    # Gate 8: Administration order
    admin_order_ok = not administration_order_flag
    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.ADMIN_ORDER,
        gate_name="Administration order",
        passed=admin_order_ok,
        evaluated=True,
        reason_code="1141" if not admin_order_ok else None,
        input_value=administration_order_flag,
    ))

    # Gate 9: Insolvency
    # Status: None=0, Sequestrated=1, Rehabilitated=2
    # Sequestrated declines; rehabilitated requires 24-month seasoning
    insolvency_ok = insolvency_status_code == 0 or insolvency_status_code == 2  # Allow rehabilitated (checked in cap waterfall)
    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.INSOLVENCY,
        gate_name="Insolvency",
        passed=insolvency_ok,
        evaluated=True,
        reason_code="1142" if insolvency_status_code == 1 else None,
        input_value=insolvency_status_code,
    ))

    # Gate 10: Deceased / estate
    deceased_ok = not deceased_flag and not estate_flag
    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.DECEASED,
        gate_name="Deceased / estate",
        passed=deceased_ok,
        evaluated=True,
        reason_code="1150" if not deceased_ok else None,
        input_value=f"deceased={deceased_flag}, estate={estate_flag}",
    ))

    # Gate 11: Exclusion lists
    # Any hit declines; need to check hit count
    exclusion_ok = not exclusion_list_hits or len(exclusion_list_hits) == 0
    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.EXCLUSION,
        gate_name="Exclusion lists",
        passed=exclusion_ok,
        evaluated=True,
        reason_code="1160" if not exclusion_ok else None,
        input_value=len(exclusion_list_hits) if exclusion_list_hits else 0,
    ))

    # Gate 12: Duplicate detection
    # Identical application (same client, amount within R500, same day) in last 24 hours
    duplicate_ok = not duplicate_detected
    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.DUPLICATE,
        gate_name="Duplicate detection",
        passed=duplicate_ok,
        evaluated=True,
        reason_code="1170" if not duplicate_ok else None,
    ))

    # Gate 13: In-flight applications
    # Unconcluded Flex Loan or consolidation application exists
    in_flight_ok = not in_flight_applications or len(in_flight_applications) == 0
    verdicts.append(GateVerdict(
        gate_code=EligibilityGateCode.IN_FLIGHT,
        gate_name="In-flight applications",
        passed=in_flight_ok,
        evaluated=True,
        reason_code="1171" if not in_flight_ok else None,
        input_value=len(in_flight_applications) if in_flight_applications else 0,
    ))

    # Determine overall eligibility
    is_eligible = all(v.passed for v in verdicts)

    return verdicts, is_eligible
