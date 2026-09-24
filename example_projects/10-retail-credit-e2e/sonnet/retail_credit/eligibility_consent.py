"""P03 -- Consent and hard eligibility (spec 10 §5.4).

Two bodies of work joined because they share a short-circuit (10 §5.4):
consent (wraps `core.consent`, mandatory bureau-enquiry consent is a hard
stop) and hard eligibility (`core.eligibility`'s frozen gates, plus two
product-10-specific gates this project owns: maximum age at maturity for
an unsecured product, and the 48-hour in-flight-application duplicate
check). **Every gate is evaluated; none short-circuits** (10 §5.4's "the
requirement is therefore: evaluate every gate, record every verdict") --
this module never returns early, matching `core.eligibility`'s own
all-gates-evaluated design.

Reused from 00: `core.eligibility.decline_reason_codes` (the nine
library-owned gates) and `core.consent.consent_verdict`. Written here: the
two product-owned gates and the combination that adds this project's own
reason codes to the library's.
"""
from __future__ import annotations

from decider import missing_as, param, step

from credit_core.consent import consent_verdict

R_MAX_AGE_AT_MATURITY = 1101
R_IN_FLIGHT_DUPLICATE = 1102
R_BUREAU_CONSENT_MISSING = 1103

UNSECURED_MAX_AGE_AT_MATURITY = 75


def product_10_extra_reasons(
    applicant_age_years: float,
    term_months: int,
    bureau_enquiry_consent: bool = missing_as(False),
    has_in_flight_application: bool = missing_as(False),
    max_age_at_maturity: float = param(75.0, ge=60.0, le=80.0),
) -> list[int]:
    """The gates 00's frozen `core.eligibility` does not carry, because they are product- and
    project-owned, not library-owned (10 §5.4: "existing-relationship requirements", "an
    in-flight-application check").
    """
    reasons = []
    age_at_maturity = applicant_age_years + term_months / 12.0
    if age_at_maturity > max_age_at_maturity:
        reasons.append(R_MAX_AGE_AT_MATURITY)
    if has_in_flight_application:
        reasons.append(R_IN_FLIGHT_DUPLICATE)
    if not bureau_enquiry_consent:
        # Bureau-enquiry consent is a hard stop, not a degradation (10 §5.4): "an enquiry
        # without consent is an offence". Recorded as a reason like any other gate, so the
        # complete reason set (10 §5.4's "short-circuit tension") still holds.
        reasons.append(R_BUREAU_CONSENT_MISSING)
    return reasons


def combine_eligibility_reasons(
    eligibility_decline_reasons: list[int], product_10_extra_reasons: list[int],
) -> list[int]:
    """Every hard-eligibility reason this decision fired, library-owned plus project-owned,
    before consent's marketing/regulated-notice classification is folded in by P18's ranking.
    """
    a = [] if eligibility_decline_reasons is None else eligibility_decline_reasons
    b = [] if product_10_extra_reasons is None else product_10_extra_reasons
    return list(a) + list(b)


product_10_extra_reasons_step = step(product_10_extra_reasons, output="product_10_extra_reasons")
combine_eligibility_reasons_step = step(combine_eligibility_reasons, output="hard_eligibility_reasons")
consent_verdict_step = step(consent_verdict, output="consent_verdict_code")

__all__ = [
    "R_MAX_AGE_AT_MATURITY", "R_IN_FLIGHT_DUPLICATE", "R_BUREAU_CONSENT_MISSING",
    "product_10_extra_reasons_step", "combine_eligibility_reasons_step", "consent_verdict_step",
]
