"""The four assessment modes. A mode picks evidence rules; it never changes the arithmetic."""
from __future__ import annotations

from decider import param

NEW_APPLICATION = 1
LIMIT_INCREASE = 2
ARRANGEMENT = 3
SCENARIO = 4


def minimum_income_tier(
    assessment_mode_code: int,
    minimum_tier_new_application: int = param(3, ge=1, le=5),
    minimum_tier_limit_increase: int = param(4, ge=1, le=5),
    minimum_tier_arrangement: int = param(5, ge=1, le=5),
    minimum_tier_scenario: int = param(3, ge=1, le=5),
) -> int:
    """The weakest income evidence the mode accepts, in `credit_core.income` tiers (1 strongest, 5 weakest).

    A new application needs a bank statement or better. A limit increase also takes a declared income, because
    it refreshes on a cadence. An arrangement takes anything. An unknown mode gets the new-application rule.
    """
    if assessment_mode_code == LIMIT_INCREASE:
        return minimum_tier_limit_increase
    if assessment_mode_code == ARRANGEMENT:
        return minimum_tier_arrangement
    if assessment_mode_code == SCENARIO:
        return minimum_tier_scenario
    return minimum_tier_new_application
