"""Hard eligibility gates. Every gate is evaluated, so the reason list is complete."""
from __future__ import annotations

from decider import allow_fallback, missing_as, param, step

# decline reason codes
R_UNDERAGE = 1001
R_NOT_RESIDENT = 1002
R_NO_CAPACITY = 1003
R_PRODUCT_UNAVAILABLE = 1004
R_EXCLUSION_LIST = 1005
R_SANCTIONED = 1006
R_DECEASED = 1007
R_DEBT_REVIEW = 1008
R_ADMINISTRATION = 1009


@allow_fallback
def decline_reason_codes(
    applicant_age_years: float,
    is_resident: bool = missing_as(True), has_contractual_capacity: bool = missing_as(True),
    product_available: bool = missing_as(True), on_exclusion_list: bool = missing_as(False),
    is_sanctioned: bool = missing_as(False), is_deceased: bool = missing_as(False),
    in_debt_review: bool = missing_as(False), in_administration: bool = missing_as(False),
    minimum_age: float = param(18.0, ge=16.0, le=21.0),
) -> list[int]:
    """The reason codes of every failed gate; empty for an eligible applicant."""
    failed = [
        (applicant_age_years < minimum_age, R_UNDERAGE), (not is_resident, R_NOT_RESIDENT),
        (not has_contractual_capacity, R_NO_CAPACITY), (not product_available, R_PRODUCT_UNAVAILABLE),
        (on_exclusion_list, R_EXCLUSION_LIST), (is_sanctioned, R_SANCTIONED), (is_deceased, R_DECEASED),
        (in_debt_review, R_DEBT_REVIEW), (in_administration, R_ADMINISTRATION),
    ]
    return [code for hit, code in failed if hit]


def is_eligible(decline_reason_codes: list[int]) -> bool:
    return len(decline_reason_codes) == 0


decline_reason_codes_step = step(decline_reason_codes)
is_eligible_step = step(is_eligible)
