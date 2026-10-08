"""Caps on the matrix's target limit. The lowest binds and is named; ties go to the lower code.

C4 (group exposure) is not modelled and C7 (affordability) is applied after the assessment, in `affordability.py`.
C2 and C3 read the income on file rather than the freshly assessed income: the assessment needs the proposed limit
these caps produce, so keying them on its output would make the two depend on each other.
"""
from __future__ import annotations

import math

from decider import flow, helper, missing_as, param, step

from limit_mgmt.vocab import PRODUCT_EVERYDAY_CARD

_ROOM = 1e-6


@helper(signatures=[((float,), float)])
def round_down_500(amount: float) -> float:
    """Down to the nearest R500, never up: rounding up could breach a cap that was computed exactly."""
    return math.floor(amount / 500.0) * 500.0


def cap_product_max(product_code: float) -> float:
    return 300_000.0 if product_code == PRODUCT_EVERYDAY_CARD else 150_000.0


def cap_income_multiple(
    declared_gross_income_on_file: float = missing_as(0.0), income_multiple: float = param(4.0, ge=1.0, le=6.0),
) -> float:
    return round(declared_gross_income_on_file * income_multiple, 2)


def cap_total_exposure(
    declared_net_income_on_file: float = missing_as(0.0), total_exposure_other: float = missing_as(0.0),
    exposure_multiple: float = param(8.0, ge=1.0, le=12.0), exposure_absolute_cap: float = param(450_000.0, ge=0.0),
) -> float:
    """The client's total unsecured limit may not pass the lower of a rand cap and a multiple of net income; this
    account may use what the client's other accounts leave."""
    return round(max(0.0, min(exposure_absolute_cap, declared_net_income_on_file * exposure_multiple)
                     - total_exposure_other), 2)


def cap_observed_spend(
    current_limit: float, observed_spend_p90: float = missing_as(0.0),
    spend_multiple: float = param(3.5, ge=0.0, le=10.0), spend_floor: float = param(2_000.0, ge=0.0),
) -> float:
    """A client whose best month in six was R400 does not get R30 000, so the limit stays where it is."""
    return round(max(current_limit, spend_multiple * observed_spend_p90, spend_floor), 2)


def policy_proposed_limit(
    uncapped_target_limit: float, current_limit: float,
    cap_product_max: float, cap_income_multiple: float, cap_total_exposure: float, cap_observed_spend: float,
    matrix_max_increase: float, matrix_min_increment_unadjusted: float = missing_as(500.0),
) -> tuple[float, int]:
    """The target held to the lowest cap, never below the current limit, rounded down to R500, and the code of the cap
    that bound (0 when the matrix's own target did). An increase under the cell's minimum increment is no increase."""
    target = max(min(uncapped_target_limit, cap_product_max, cap_income_multiple, cap_total_exposure,
                     cap_observed_spend, current_limit + matrix_max_increase), current_limit)
    binding = 0
    if abs(cap_product_max - target) < _ROOM:
        binding = 1
    elif abs(cap_income_multiple - target) < _ROOM:
        binding = 2
    elif abs(cap_total_exposure - target) < _ROOM:
        binding = 3
    elif abs(cap_observed_spend - target) < _ROOM:
        binding = 5
    elif abs(current_limit + matrix_max_increase - target) < _ROOM:
        binding = 6
    rounded = round_down_500(target)
    return (current_limit if rounded - current_limit < matrix_min_increment_unadjusted else rounded), binding


def policy_proposed_limit_unadjusted(
    uncapped_target_limit_unadjusted: float, current_limit: float,
    cap_product_max: float, cap_income_multiple: float, cap_total_exposure: float, cap_observed_spend: float,
    matrix_max_increase_unadjusted: float, matrix_min_increment_unadjusted: float = missing_as(500.0),
) -> float:
    """The same arithmetic on the cell as authored, with every overlay off."""
    target = max(min(uncapped_target_limit_unadjusted, cap_product_max, cap_income_multiple, cap_total_exposure,
                     cap_observed_spend, current_limit + matrix_max_increase_unadjusted), current_limit)
    rounded = round_down_500(target)
    return current_limit if rounded - current_limit < matrix_min_increment_unadjusted else rounded


unit = flow(
    step(cap_product_max), step(cap_income_multiple), step(cap_total_exposure), step(cap_observed_spend),
    step(policy_proposed_limit, outputs=("policy_proposed_limit", "policy_binding_cap_code")),
    step(policy_proposed_limit_unadjusted), name="caps")
