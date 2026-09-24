"""§5.7 -- the decrease path, one trigger (SCOPE.md: "the decrease path
except one trigger"). D03 (emerging arrears) is chosen: it is
Immediate-class, so it also exercises the simultaneous-case precedence
rule (§5.7: "an immediate-class decrease... suppresses every increase for
that client in the cycle") without needing the notice-period machinery
(§6.7) that a Notice-class trigger would pull in -- explicitly out of
scope here ("notice and consent mechanics beyond recording").

Decreases are not subject to §5.2's exclusions or the consent requirement
(§5.7: "a reduction is a unilateral act the Bank is entitled to take with
reasons and notice") -- this module runs unconditionally on every account,
independent of `exclusions.py`.
"""
from __future__ import annotations

from decider import missing_as, step

from limit_mgmt.vocab import NoticeClass

D03_EMERGING_ARREARS = "D03"


def decrease_trigger_fires(
    worst_arrears_months_now: int = missing_as(0),
) -> bool:
    """D03 (§5.7): "1 cycle past due in the current month"."""
    return bool(worst_arrears_months_now and worst_arrears_months_now >= 1)


def decrease_target_limit(
    decrease_trigger_fires: bool,
    current_limit: float,
    statement_balance: float = missing_as(0.0),
    unsettled_authorisations: float = missing_as(0.0),
    accrued_unbilled_interest: float = missing_as(0.0),
) -> float:
    """D03 has no reduction rule of its own named in the spec's table beyond the
    universal floor (rule 1, §5.7): "the new limit must be at least the statement
    balance plus unsettled authorisations plus accrued unbilled interest." Working
    depth: D03 reduces to that floor directly (the most conservative reduction any
    trigger in the table produces), rather than adding a second, trigger-specific
    target on top -- SCOPE.md's "one trigger" is read as one *complete* decrease
    path (trigger, floor, notice class, precedence), not one trigger plus a bespoke
    target formula only D03 would use."""
    if not decrease_trigger_fires:
        return current_limit
    floor = statement_balance + unsettled_authorisations + accrued_unbilled_interest
    return round(min(current_limit, max(floor, 0.0)), 2)


def decrease_notice_class(decrease_trigger_fires: bool) -> int:
    return NoticeClass.IMMEDIATE if decrease_trigger_fires else NoticeClass.NOT_APPLICABLE


def decrease_trigger_codes(decrease_trigger_fires: bool) -> list[str]:
    return [D03_EMERGING_ARREARS] if decrease_trigger_fires else []


decrease_trigger_fires_step = step(decrease_trigger_fires)
decrease_target_limit_step = step(decrease_target_limit, output="decrease_target_limit")
decrease_notice_class_step = step(decrease_notice_class, output="decrease_notice_class_code")
decrease_trigger_codes_step = step(decrease_trigger_codes, output="decrease_trigger_codes")
