"""The decrease path, with one trigger: D03, emerging arrears (one cycle past due this month).

D03 is immediate-class, so it also exercises the rule that a decrease on one account suppresses increases on the
client's others. The other thirteen triggers and the notice machinery are left out.
"""
from __future__ import annotations

from decider import allow_fallback, flow, missing_as, step

from limit_mgmt.vocab import NoticeClass


def decrease_trigger_fires(worst_arrears_months_now: int = missing_as(0)) -> bool:
    return worst_arrears_months_now >= 1


def decrease_target_limit(
    decrease_trigger_fires: bool, current_limit: float, statement_balance: float = missing_as(0.0),
    unsettled_authorisations: float = missing_as(0.0), accrued_unbilled_interest: float = missing_as(0.0),
) -> float:
    """The lowest limit allowed: the statement balance plus unsettled authorisations and accrued interest, so a
    transaction already authorised cannot put the client over limit. Never above the current limit."""
    if not decrease_trigger_fires:
        return current_limit
    return round(min(current_limit, statement_balance + unsettled_authorisations + accrued_unbilled_interest), 2)


def decrease_notice_class_code(decrease_trigger_fires: bool) -> int:
    return NoticeClass.IMMEDIATE if decrease_trigger_fires else NoticeClass.NOT_APPLICABLE


@allow_fallback
def decrease_trigger_codes(decrease_trigger_fires: bool) -> list[str]:
    return ["D03"] if decrease_trigger_fires else []


unit = flow(step(decrease_trigger_fires), step(decrease_target_limit), step(decrease_notice_class_code),
            step(decrease_trigger_codes), name="decrease")
