"""`core.eligibility` -- hard eligibility gates (spec 00 §6.15). Thin, frozen interface.

Age, residency, capacity to contract, product availability, existing
relationship status, exclusion lists, sanctioned status, deceased/estate
flags, debt review and administration status. Every gate is evaluated (not
only until the first failure), so the reason set is complete even though
the outcome short-circuits downstream scoring (00 §6.15 "Hard part").
"""
from __future__ import annotations

from decider import missing_as, param, step

# decline_reason_codes for eligibility gates
R_UNDERAGE = 1001
R_NOT_RESIDENT = 1002
R_NO_CAPACITY = 1003
R_PRODUCT_UNAVAILABLE = 1004
R_EXCLUSION_LIST = 1005
R_SANCTIONED = 1006
R_DECEASED = 1007
R_DEBT_REVIEW = 1008
R_ADMINISTRATION = 1009


def decline_reason_codes(
    applicant_age_years: float, is_resident: bool = missing_as(True), has_contractual_capacity: bool = missing_as(True),
    product_available: bool = missing_as(True), on_exclusion_list: bool = missing_as(False),
    is_sanctioned: bool = missing_as(False), is_deceased: bool = missing_as(False),
    in_debt_review: bool = missing_as(False), in_administration: bool = missing_as(False),
    minimum_age: float = param(18.0, ge=16.0, le=21.0),
) -> list[int]:
    reasons = []
    if applicant_age_years < minimum_age:
        reasons.append(R_UNDERAGE)
    if not is_resident:
        reasons.append(R_NOT_RESIDENT)
    if not has_contractual_capacity:
        reasons.append(R_NO_CAPACITY)
    if not product_available:
        reasons.append(R_PRODUCT_UNAVAILABLE)
    if on_exclusion_list:
        reasons.append(R_EXCLUSION_LIST)
    if is_sanctioned:
        reasons.append(R_SANCTIONED)
    if is_deceased:
        reasons.append(R_DECEASED)
    if in_debt_review:
        reasons.append(R_DEBT_REVIEW)
    if in_administration:
        reasons.append(R_ADMINISTRATION)
    return reasons


def is_eligible(decline_reason_codes: list[int]) -> bool:
    return len(decline_reason_codes) == 0


decline_reason_codes_step = step(decline_reason_codes)
is_eligible_step = step(is_eligible)
