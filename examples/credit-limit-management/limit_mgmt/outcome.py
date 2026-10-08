"""The per-account answer: the limit to move to, the kind of change, the notice class and why there is no increase.

Whether an increase is actually funded is decided later, over the whole book, in `allocation.py`.
"""
from __future__ import annotations

from decider import allow_fallback, flow, step

from limit_mgmt import reasons
from limit_mgmt.vocab import ChangeType, IncreasePath, NoticeClass


def final_proposed_limit(
    is_excluded: bool, increase_path_code: int, cap_affordability: float, current_limit: float,
    decrease_trigger_fires: bool, decrease_target_limit: float,
) -> float:
    """The decrease target when a trigger fired, else the affordability-capped proposal; no change when the account is
    excluded or the increase is not offered. Recorded for every account."""
    if decrease_trigger_fires:
        return decrease_target_limit
    if is_excluded or increase_path_code == IncreasePath.FAIL:
        return current_limit
    return cap_affordability


def change_type_code(
    is_excluded: bool, increase_path_code: int, decrease_trigger_fires: bool, final_proposed_limit: float,
    current_limit: float,
) -> int:
    if decrease_trigger_fires:
        return ChangeType.DECREASE
    if is_excluded or final_proposed_limit <= current_limit:
        return ChangeType.NO_CHANGE
    return ChangeType.INCREASE_AUTOMATIC if increase_path_code == IncreasePath.AUTOMATIC else ChangeType.INCREASE_CONDITIONAL


def notice_class_code(change_type_code: int, decrease_notice_class_code: int) -> int:
    """An accepted increase takes effect at once; a conditional one needs the client's confirmation first."""
    if change_type_code == ChangeType.DECREASE:
        return decrease_notice_class_code
    if change_type_code == ChangeType.INCREASE_AUTOMATIC:
        return NoticeClass.IMMEDIATE
    if change_type_code == ChangeType.INCREASE_CONDITIONAL:
        return NoticeClass.CONSENT_REQUIRED
    return NoticeClass.NOT_APPLICABLE


@allow_fallback
def decline_reason_codes(
    is_excluded: bool, increase_path_code: int, current_limit: float, policy_proposed_limit: float,
    decrease_trigger_fires: bool,
) -> list[int]:
    """Why this account has no increase, before the budget: excluded, affordability failed, or the matrix and caps left none."""
    if decrease_trigger_fires:
        return []
    if is_excluded:
        return [reasons.R_EXCLUDED]
    codes = [reasons.R_AFFORDABILITY_FAIL] if increase_path_code == IncreasePath.FAIL else []
    if policy_proposed_limit <= current_limit:
        codes.append(reasons.R_NO_INCREASE_AFTER_CAPS)
    return codes


unit = flow(step(final_proposed_limit), step(change_type_code), step(notice_class_code), step(decline_reason_codes),
            reasons.REASON_REGISTRY.resolve_step(), name="outcome")
