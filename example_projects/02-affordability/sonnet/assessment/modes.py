"""The four assessment modes (spec 02 §5.8).

"A mode may not change the arithmetic. It selects evidence rules, parameter
sets and which outputs are produced." (§5.8) -- enforced here structurally,
not by convention: every mode runs through the one `pipeline.build()` dag;
`assessment_mode_code` only ever selects which *parameter* applies
(`minimum_income_tier` below) or, in `household.merge_household_accounts`,
how the account list is shaped before the one shared `core.obligations` unit
runs. No mode has its own copy of a stage.
"""
from __future__ import annotations

from decider import param

NEW_APPLICATION = 1
LIMIT_INCREASE = 2
ARRANGEMENT = 3
SCENARIO = 4

MODE_NAMES = {
    NEW_APPLICATION: "new_application", LIMIT_INCREASE: "limit_increase",
    ARRANGEMENT: "arrangement", SCENARIO: "scenario",
}


def minimum_income_tier(
    assessment_mode_code: int,
    minimum_tier_new_application: int = param(4, ge=1, le=6),
    minimum_tier_limit_increase: int = param(3, ge=1, le=6),
    minimum_tier_arrangement: int = param(6, ge=1, le=6),
    minimum_tier_scenario: int = param(4, ge=1, le=6),
) -> int:
    """§5.2.1: "the weakest permitted tier is a product parameter" -- this slice keys it on
    mode instead (02's own axis of variation), matching §5.8's table: a limit increase
    accepts weaker evidence (tier 3, internal deposits) than a new application (tier 4);
    an arrangement, already in force, accepts the weakest (tier 6, "often unverifiable")."""
    return {
        NEW_APPLICATION: minimum_tier_new_application,
        LIMIT_INCREASE: minimum_tier_limit_increase,
        ARRANGEMENT: minimum_tier_arrangement,
        SCENARIO: minimum_tier_scenario,
    }.get(assessment_mode_code, minimum_tier_new_application)
