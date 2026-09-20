"""Affordability assessment — discretionary income and capacity.

Published interface:
- Inputs: net_monthly_income, living_expenses, existing_obligations, instalment
- Outputs: discretionary_income, max_affordable_instalment, affordability_verdict_code
- Params: AffordabilityParams

Implements: Credit Policy §6.5
"""

from decider2 import module

from . import steps
from .params import AffordabilityParams

Affordability = module(
    steps.living_expenses,
    steps.existing_obligations,
    steps.discretionary_income,
    steps.affordability_verdict_code,
    steps.max_affordable_instalment,
    name="affordability",
    params=AffordabilityParams,
)

__all__ = ["Affordability", "AffordabilityParams"]
