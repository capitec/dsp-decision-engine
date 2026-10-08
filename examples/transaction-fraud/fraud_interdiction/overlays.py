"""The overlay stack: approved, dated, scoped changes layered on the rule set without editing it."""
from __future__ import annotations

from datetime import date

from decider import allow_fallback, param, step

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister

from fraud_interdiction.vocab import CHANNEL_WEB, Action

ADJUSTMENT_SET_ID = "ADJS-2026.01"
AMOUNT_MULTIPLIER = "mule_scam_amount_multiplier"
AT_WEB_ACTION = "at_web_action"

REGISTER = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-2026-001", kind="cap_adjustment", target=AMOUNT_MULTIPLIER,
        effect=AdjustmentEffect("multiply", round(1 / 1.4, 4)), scope={"family": "MS"}, stack_position=1,
        owner="24/7 duty officer", approval_reference="DUTY-2026-0311",
        rationale="Mule and scam family at 1.4x sensitivity for 72 hours during a live scam wave",
        effective_from=date(2026, 3, 11), effective_to=date(2026, 3, 14), review_date=date(2026, 3, 14),
        tighten_only=True),
    Adjustment(
        adjustment_id="ADJ-2026-002", kind="cap_adjustment", target=AMOUNT_MULTIPLIER,
        effect=AdjustmentEffect("multiply", 0.7), scope={"family": "MS"}, stack_position=2,
        owner="Head of Fraud", approval_reference="HOF-2025-1122",
        rationale="Mule and scam amount thresholds x0.7 for the festive period",
        effective_from=date(2025, 12, 1), effective_to=date(2026, 1, 15), review_date=date(2026, 1, 8),
        tighten_only=True),
    Adjustment(
        adjustment_id="ADJ-2026-003", kind="action_escalation", target=AT_WEB_ACTION,
        effect=AdjustmentEffect("set", float(Action.STEP_UP)), scope={"family": "AT", "channel_code": CHANNEL_WEB},
        stack_position=1, owner="Head of Fraud", approval_reference="HOF-2026-0044",
        rationale="Account-takeover rules that only monitor ask for a step-up on the web channel",
        effective_from=date(2026, 1, 1), effective_to=date(2027, 1, 1), review_date=date(2026, 10, 1)),
])


def escalated_action(action: int, family: str, channel_code: int, decision_date: date, stack_enabled: bool) -> int:
    """The action a rule asks for once the escalation overlay (monitor to step-up) is applied."""
    if action != Action.MONITOR:
        return action
    result = REGISTER.apply_stack(AT_WEB_ACTION, float(action), {"family": family, "channel_code": channel_code},
                                  decision_date, ADJUSTMENT_SET_ID, stack_enabled=stack_enabled)
    return int(result.adjusted_value)


def adjustment_stack_enabled(adjustment_stack_enabled: bool = param(True)) -> bool:
    """Policy switch, shown as a column so every step that applies overlays reads the same one."""
    return adjustment_stack_enabled


@allow_fallback
def amount_overlays(decision_date: date, adjustment_stack_enabled: bool) -> tuple[float, float, str, list[str]]:
    result = REGISTER.apply_stack(AMOUNT_MULTIPLIER, 1.0, {"family": "MS"}, decision_date, ADJUSTMENT_SET_ID,
                                  stack_enabled=adjustment_stack_enabled)
    return (result.adjusted_value, result.unadjusted_value, result.adjustment_set_id,
            list(result.adjustments_applied))


multiplier_step = step(amount_overlays, outputs=(AMOUNT_MULTIPLIER, f"{AMOUNT_MULTIPLIER}_unadjusted",
                                             "adjustment_set_id", "applied_adjustment_ids"))
