"""Hard eligibility: the shared library's gates plus three of this product's. Every gate is checked, none short-circuits."""
from __future__ import annotations

from decider import allow_fallback, missing_as, param, step

from credit_core import eligibility

R_AGE_AT_MATURITY = 1101
R_IN_FLIGHT_DUPLICATE = 1102
R_BUREAU_CONSENT_MISSING = 1103


@allow_fallback
def hard_eligibility_reasons(
    eligibility_decline_reasons: list[int], applicant_age_years: float, term_months: int,
    bureau_enquiry_consent: bool = missing_as(False), has_in_flight_application: bool = missing_as(False),
    max_age_at_maturity: float = param(75.0, ge=60.0, le=80.0),
) -> list[int]:
    """The library's failed gates, then: too old at maturity, a duplicate application in flight, no bureau consent.

    An enquiry without consent is an offence, so missing consent is a hard stop like any other gate.
    """
    extra = [
        (applicant_age_years + term_months / 12.0 > max_age_at_maturity, R_AGE_AT_MATURITY),
        (has_in_flight_application, R_IN_FLIGHT_DUPLICATE),
        (not bureau_enquiry_consent, R_BUREAU_CONSENT_MISSING),
    ]
    return [*eligibility_decline_reasons, *(code for hit, code in extra if hit)]


eligibility_gates_step = step(eligibility.decline_reason_codes, output="eligibility_decline_reasons")
hard_eligibility_reasons_step = step(hard_eligibility_reasons)
is_eligible_step = eligibility.is_eligible_step.relabel(reads={"decline_reason_codes": "hard_eligibility_reasons"})
