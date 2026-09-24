"""Final per-account assembly: `final_proposed_limit`, `change_type_code`,
`notice_class_code`, `decline_reason_codes`. This is the last stage of the
**per-account** pipeline (§5.1-§5.7, §5.9 minus notice/consent mechanics) --
everything before §5.8, the population-level stage `allocation.py` owns.
"""
from __future__ import annotations

from decider import missing_as, step

from limit_mgmt import reasons
from limit_mgmt.vocab import ChangeType, IncreasePath, NoticeClass


def final_proposed_limit(
    is_excluded: bool, increase_path_code: int, cap_affordability: float, current_limit: float,
) -> float:
    """§7: recorded for every account, including excluded ones -- `current_limit`
    (no change) when excluded or when affordability fails, else the affordability-capped
    policy proposal (C7, §5.5)."""
    if is_excluded or increase_path_code == IncreasePath.FAIL:
        return current_limit
    return cap_affordability


final_proposed_limit_step = step(final_proposed_limit)


def change_type_code(
    is_excluded: bool, increase_path_code: int, decrease_trigger_fires: bool,
    final_proposed_limit: float, current_limit: float,
) -> int:
    if decrease_trigger_fires:
        return ChangeType.DECREASE
    if is_excluded or final_proposed_limit <= current_limit:
        return ChangeType.NO_CHANGE
    if increase_path_code == IncreasePath.AUTOMATIC:
        return ChangeType.INCREASE_AUTOMATIC
    if increase_path_code == IncreasePath.CONDITIONAL:
        return ChangeType.INCREASE_CONDITIONAL
    return ChangeType.NO_CHANGE


def notice_class_code(change_type_code: int, decrease_notice_class_code: int = missing_as(NoticeClass.NOT_APPLICABLE)) -> int:
    if change_type_code == ChangeType.DECREASE:
        return decrease_notice_class_code
    if change_type_code == ChangeType.INCREASE_AUTOMATIC:
        return NoticeClass.IMMEDIATE  # §6.7: "Increase, offer accepted | Immediate"
    if change_type_code == ChangeType.INCREASE_CONDITIONAL:
        return NoticeClass.CONSENT_REQUIRED
    return NoticeClass.NOT_APPLICABLE


change_type_code_step = step(change_type_code)
notice_class_code_step = step(notice_class_code)


def decline_reason_codes(
    is_excluded: bool, exclusion_codes: list[int] = missing_as([]),
    increase_path_code: int = missing_as(IncreasePath.FAIL),
    final_proposed_limit: float = missing_as(0.0), current_limit: float = missing_as(0.0),
    matrix_max_increase: float = missing_as(0.0),
) -> list[int]:
    """Per-account decline reasons -- distinct from allocation.py's non-selection
    reasons (`R_BELOW_FUNDING_LINE` etc.), which need the whole population and are
    attached after `run_allocation()` (§5.8 "Recorded": the client-facing answer names
    the rank and the line, which a per-account step cannot know)."""
    codes = []
    if is_excluded:
        codes.append(reasons.R_EXCLUDED)
        return codes
    if increase_path_code == IncreasePath.FAIL:
        codes.append(reasons.R_AFFORDABILITY_FAIL)
    if final_proposed_limit <= current_limit and matrix_max_increase <= 0:
        codes.append(reasons.R_MATRIX_ZERO_OR_BELOW_MINIMUM_INCREMENT)
    return codes


decline_reason_codes_step = step(decline_reason_codes, output="decline_reason_codes")
