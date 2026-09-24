"""§5.5 -- caps. Six of the spec's seven bind before affordability; C7
(affordability) is folded in by `affordability.py` once the assessment
returns, since it depends on the assessment's own output. The five
policy-ground caps here (C1, C2, C3, C5, C6) are evaluated independently --
"every cap's computed value must be available" (§5.5 "Recorded") -- and the
lowest wins, ties broken by this module's table order (deterministic,
per §5.5).

Rounding is this project's own (`round_down_500`), not `core.rounding`'s
`round_advance` (nearest R100): §5.5 requires **down**, without exception
("rounding up can breach a cap that was computed exactly"), which
`core.rounding.round_advance` (round-to-nearest) does not guarantee -- a
genuine, small, project-specific rule, not a fork of the library's.
"""
from __future__ import annotations

from decider import missing_as, param, step

from limit_mgmt.vocab import Cap, NO_CAP_BINDING, PRODUCT_ACCESS_FACILITY, PRODUCT_EVERYDAY_CARD

_PRODUCT_MAX = {PRODUCT_EVERYDAY_CARD: 300_000.0, PRODUCT_ACCESS_FACILITY: 150_000.0}


def round_down_500(amount: float) -> float:
    """§5.5: "rounded down to the nearest R500... a cap that can be breached by
    rounding is not a cap." """
    import math
    return math.floor(amount / 500.0) * 500.0


def cap_product_max(product_code: float) -> float:
    return _PRODUCT_MAX.get(product_code, 150_000.0)


def cap_income_multiple(
    declared_gross_income_on_file: float = missing_as(0.0),
    income_multiple: float = param(4.0, ge=1.0, le=6.0, description="§6.4: 2.5-5.0 by product/segment"),
) -> float:
    """§5.5 C2, §6.4. Keyed on `declared_gross_income_on_file` for the same reason as C3
    (see `cap_total_exposure`'s docstring): caps precede affordability's own
    freshly-assessed `gross_monthly_income` in the spec's stage order."""
    return round(declared_gross_income_on_file * income_multiple, 2)


def cap_total_exposure(
    declared_net_income_on_file: float = missing_as(0.0),
    total_exposure_other: float = missing_as(0.0),
    exposure_multiple: float = param(8.0, ge=1.0, le=12.0),
    exposure_absolute_cap: float = param(450_000.0, ge=0.0),
) -> float:
    """§5.5 C3: the client's aggregate unsecured limit across the Bank <= the lower of a
    rand cap and a multiple of net income. Deliberately keyed on
    `declared_net_income_on_file` (a raw account-state figure captured at origination or
    last refresh), **not** the freshly-assessed `net_monthly_income` §5.6 produces: caps
    (§5.5) precede affordability (§5.6) in the spec's own stage order, so a cap that
    needed the affordability module's *output* would make caps and affordability
    circularly dependent on each other -- `decider` catches exactly this
    (`WiringError: ['caps', 'affordability_bridge', 'outcome'] depend on each other in a
    cycle`) if `net_monthly_income` is used here instead. See NOTES.md "Framework
    friction" and "Spec problems". `total_exposure_other` is the client's exposure on
    every *other* account (this account's own current limit is added back by the
    caller, since the cap governs the total, not the increment)."""
    by_income = declared_net_income_on_file * exposure_multiple
    return round(min(exposure_absolute_cap, by_income), 2)


def cap_observed_spend(
    current_limit: float, observed_spend_p90: float = missing_as(0.0),
    spend_multiple: float = param(3.5, ge=0.0, le=10.0),
    spend_floor: float = param(2_000.0, ge=0.0),
) -> float:
    """§5.5 C5: "the cap that stops the programme doing something indefensible." """
    return round(max(current_limit, spend_multiple * observed_spend_p90, spend_floor), 2)


cap_product_max_step = step(cap_product_max, output="cap_product_max")
cap_income_multiple_step = step(cap_income_multiple, output="cap_income_multiple")
cap_total_exposure_step = step(cap_total_exposure, output="cap_total_exposure")
cap_observed_spend_step = step(cap_observed_spend, output="cap_observed_spend")


def policy_proposed_limit(
    uncapped_target_limit: float, current_limit: float,
    cap_product_max: float, cap_income_multiple: float, cap_total_exposure: float, cap_observed_spend: float,
    matrix_max_increase: float, matrix_min_increment_unadjusted: float = missing_as(500.0),
) -> tuple[float, int]:
    """The five policy-ground caps (C1, C2, C3, C5, C6), lowest wins, deterministic tie
    break by this table's own order (§5.5). C7 (affordability) is applied afterwards by
    `affordability.py`, since it needs this stage's output as its own input (the
    notional instalment is a function of `proposed_limit`)."""
    matrix_ceiling = current_limit + matrix_max_increase
    candidates = [
        (Cap.C1_PRODUCT_MAX, cap_product_max),
        (Cap.C2_INCOME_MULTIPLE, cap_income_multiple),
        (Cap.C3_TOTAL_EXPOSURE, cap_total_exposure),
        (Cap.C5_OBSERVED_SPEND, cap_observed_spend),
        (Cap.C6_MATRIX_MAX_INCREASE, matrix_ceiling),
    ]
    target = min(uncapped_target_limit, *(v for _, v in candidates))
    target = max(target, current_limit)
    rounded = round_down_500(target)
    binding = NO_CAP_BINDING
    for code, value in candidates:
        if abs(value - target) < 1e-6:
            binding = int(code)
            break
    # §5.4: below the cell's minimum increment, the cell yields no increase at all.
    if rounded - current_limit < matrix_min_increment_unadjusted:
        rounded = current_limit
    return rounded, binding


policy_proposed_limit_step = step(policy_proposed_limit, outputs=("policy_proposed_limit", "binding_cap_code"))


def policy_proposed_limit_unadjusted(
    uncapped_target_limit_unadjusted: float, current_limit: float,
    cap_product_max: float, cap_income_multiple: float, cap_total_exposure: float, cap_observed_spend: float,
    matrix_max_increase_unadjusted: float,
) -> float:
    """Same arithmetic, overlay stack disabled -- must survive beside the adjusted
    figure everywhere (§5.4 "Emits", §5.5)."""
    matrix_ceiling = current_limit + matrix_max_increase_unadjusted
    target = min(uncapped_target_limit_unadjusted, cap_product_max, cap_income_multiple,
                 cap_total_exposure, cap_observed_spend, matrix_ceiling)
    return round_down_500(max(target, current_limit))


policy_proposed_limit_unadjusted_step = step(policy_proposed_limit_unadjusted, output="policy_proposed_limit_unadjusted")
