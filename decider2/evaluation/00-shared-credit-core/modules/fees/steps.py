"""Fee calculation capability.

Implements: Credit Policy §6.7
"""


def initiation_fee(offered_amount: float, product_code: int, params) -> float:
    """One-off initiation fee, statutorily capped.

    Simplified: 1.5% of offered amount for all products, capped at R2000.
    Implements: Credit Policy §6.7.1
    """
    fee = offered_amount * params.initiation_fee_rate
    return min(fee, params.initiation_fee_cap)


def monthly_service_fee(offered_amount: float, params) -> float:
    """Monthly service fee, statutorily capped.

    Simplified: 0.5% of offered amount per month, capped at R150.
    Implements: Credit Policy §6.7.2
    """
    fee = offered_amount * params.monthly_service_fee_rate
    return min(fee, params.monthly_service_fee_cap)


def credit_life_premium(
    offered_amount: float,
    applicant_age_years: float,
    employment_type_code: int,
    params,
) -> float:
    """Monthly credit life insurance premium.

    Simplified: R0.50 per R1000 of cover, adjusted for age.
    Implements: Credit Policy §6.8
    """
    if applicant_age_years > params.max_age_for_cover:
        return 0.0

    base_rate = params.base_premium_rate
    if applicant_age_years > 60:
        base_rate *= params.senior_age_multiplier
    elif applicant_age_years < 25:
        base_rate *= params.youth_age_multiplier

    return (offered_amount / 1000.0) * base_rate
