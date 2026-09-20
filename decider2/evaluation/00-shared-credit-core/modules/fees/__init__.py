"""Fee calculation — statutory fees and insurance premiums.

Published interface:
- Inputs: offered_amount, product_code, applicant_age_years, employment_type_code
- Outputs: initiation_fee, monthly_service_fee, credit_life_premium
- Params: FeesParams

Implements: Credit Policy §6.7 and §6.8
"""

from decider2 import module

from . import steps
from .params import FeesParams

Fees = module(
    steps.initiation_fee,
    steps.monthly_service_fee,
    steps.credit_life_premium,
    name="fees",
    params=FeesParams,
)

__all__ = ["Fees", "FeesParams"]
