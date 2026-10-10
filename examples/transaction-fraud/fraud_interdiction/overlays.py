"""The overlay stack: approved, dated, scoped changes layered on the rule set without editing it."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from decikit import ConfigurableStep, allow_fallback, param

from credit_core.adjustments import AdjustmentStack, evaluate

from fraud_interdiction.vocab import Action

AMOUNT_MULTIPLIER = "mule_scam_amount_multiplier"
AT_WEB_ACTION = "at_web_action"


def escalated_action(action: int, overlays: tuple, family: str, channel_code: int, on: int, stack_enabled: bool) -> int:
    """The action a rule asks for once the escalation overlays (monitor to step-up) are applied.

    `overlays` is the escalation stack as `AdjustmentStack` packs it; `on` is the decision date's ordinal.
    """
    if action != Action.MONITOR:
        return action
    value, _, _ = evaluate(overlays, float(action), {"family": family, "channel_code": channel_code}, on, stack_enabled)
    return int(value)


def adjustment_stack_enabled(adjustment_stack_enabled: bool = param(True)) -> bool:
    """Policy switch, shown as a column so every step that applies overlays reads the same one."""
    return adjustment_stack_enabled


@allow_fallback
def apply_amount_overlays(
    decision_date: date, adjustment_stack_enabled: bool, overlays: tuple, set_id: str, family: str,
) -> tuple[float, float, str, list[str]]:
    value, applied, _ = evaluate(overlays, 1.0, {"family": family}, decision_date.toordinal(), adjustment_stack_enabled)
    return value, 1.0, set_id, applied


class AmountOverlays(ConfigurableStep):
    """The overlays on the mule and scam amount thresholds, as a multiplier starting at 1.

    The `stack` is an `adjustment_stack` document whose overlays are scoped to `family`. The step reads the
    `adjustment_stack_enabled` column rather than a param of its own, so one switch turns every overlay off.
    It writes `mule_scam_amount_multiplier`, `mule_scam_amount_multiplier_unadjusted`, `adjustment_set_id` and
    `applied_adjustment_ids`.

    Example::

        overlays = ConfigurableStep.load("configs/0.1.0/amount_overlays.json")
        flow(adjustment_stack_enabled, overlays).score({"decision_date": date(2025, 12, 15)})
    """

    type: Literal["amount_overlays"] = "amount_overlays"
    family: str = "MS"
    stack: AdjustmentStack

    def to_ir(self, ctx: Any) -> Any:
        return ctx.call(
            self, apply_amount_overlays,
            values={"overlays": self.stack.packed(), "set_id": self.stack.set_id, "family": self.family},
            outputs=(AMOUNT_MULTIPLIER, f"{AMOUNT_MULTIPLIER}_unadjusted", "adjustment_set_id", "applied_adjustment_ids"))
