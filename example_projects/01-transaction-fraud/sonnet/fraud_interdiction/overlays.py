"""The adjustment overlay stack (spec 01 §5.9, §6.5) -- three of the seven kinds (SCOPE.md).

Built entirely on `credit_core.adjustments` (DEPS.md: 01's one hard, named
reuse from 00). 00's own `AdjustmentRegister.apply_stack_step` convenience
wrapper is scoped to *credit's* four provenance keys (`product_code`,
`segment_code`, `channel_code`, `scorecard_id`) -- its own docstring
invites a consumer with a different scope to call `apply_stack` directly,
which is what this module does: fraud's provenance is `family` and
`channel_code`, not a product or a scorecard.

Overlay kinds built (SCOPE.md: "a sensitivity dial, a threshold multiplier
and action escalation"), both on the mule/scam family -- the two literal
examples from spec §6.5's own table, stacked in declared composition
order to demonstrate that order is part of the definition (§6.5 property
3), plus the account-takeover action escalation:

1. **Sensitivity dial** (`ADJ-2026-001`, stack position 1): mule/scam family
   run 1.4x more sensitive for 72 hours during a live attack (§11 scenario
   2) -- expressed as its inverse on the amount threshold multiplier
   (lower threshold = more sensitive).
2. **Threshold multiplier** (`ADJ-2026-002`, stack position 2): mule/scam
   amount thresholds x0.7 for the festive period (§6.5's own example,
   verbatim).
3. **Action escalation** (`ADJ-2026-003`): account-takeover rules'
   `monitor` action escalates to `step_up` on the web channel (§6.5's own
   example, verbatim).

`fraud_interdiction.rules` tags the mule/scam rules whose amount threshold
reads this stack's effective value (`overlay_eligible=True`,
`base_condition` set) -- see `rules.MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE`.
"""
from __future__ import annotations

from datetime import date

from decider import frame_step, param, step
import polars as pl

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister

from fraud_interdiction import vocab
from fraud_interdiction.rules import MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE

ADJUSTMENT_SET_ID = "ADJS-2026.01"

OVERLAY_REGISTER = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-2026-001", kind="cap_adjustment", target=MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE,
        effect=AdjustmentEffect("multiply", round(1.0 / 1.4, 4)), scope={"family": vocab.FAMILY_MULE_SCAM},
        stack_position=1, owner="24/7 duty officer", approval_reference="DUTY-2026-0311",
        rationale="Mule/scam family run 1.4x more sensitive for 72h during a live scam wave (§11 scenario 2)",
        effective_from=date(2026, 3, 11), effective_to=date(2026, 3, 14), review_date=date(2026, 3, 14),
        tighten_only=True,
    ),
    Adjustment(
        adjustment_id="ADJ-2026-002", kind="cap_adjustment", target=MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE,
        effect=AdjustmentEffect("multiply", 0.7), scope={"family": vocab.FAMILY_MULE_SCAM},
        stack_position=2, owner="Head of Fraud", approval_reference="HOF-2025-1122",
        rationale="Mule/scam amount thresholds x0.7 for the festive period (§6.5 example, verbatim)",
        effective_from=date(2025, 12, 1), effective_to=date(2026, 1, 15), review_date=date(2026, 1, 8),
        tighten_only=True,
    ),
    Adjustment(
        adjustment_id="ADJ-2026-003", kind="action_escalation", target="at_web_action",
        effect=AdjustmentEffect("set", float(vocab.ACTION_STEP_UP)), scope={"family": vocab.FAMILY_ACCOUNT_TAKEOVER,
                                                                             "channel_code": 4},
        stack_position=1, owner="Head of Fraud", approval_reference="HOF-2026-0044",
        rationale="monitor -> step_up for account-takeover rules on the web channel only (§6.5 example, verbatim)",
        effective_from=date(2026, 1, 1), effective_to=date(2027, 1, 1), review_date=date(2026, 10, 1),
    ),
])


def _stack_enabled(adjustment_stack_enabled: bool = param(True)) -> bool:
    """Exposes the stack on/off toggle as a *column* (§7.6, acceptance §10 item 8: "same code
    path, one flag flipped"). `frame_step` functions receive only the `DataFrame` -- there is no
    `param()` support inside one (see NOTES.md "Framework friction") -- so this tiny scalar `step`
    materialises the toggle as a column every downstream `frame_step` can read like any other."""
    return adjustment_stack_enabled


adjustment_stack_enabled_step = step(_stack_enabled, output="adjustment_stack_enabled")


def effective_threshold_multiplier(
    decision_date: date, family: str = vocab.FAMILY_MULE_SCAM,
    adjustment_stack_enabled: bool = True,
) -> tuple[float, float, str, list]:
    """The mule/scam amount-threshold multiplier: base 1.0, composed with any in-scope, in-force overlay."""
    result = OVERLAY_REGISTER.apply_stack(
        MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE, 1.0, {"family": family}, decision_date,
        ADJUSTMENT_SET_ID, stack_enabled=adjustment_stack_enabled,
    )
    return (result.adjusted_value, result.unadjusted_value, result.adjustment_set_id,
            list(result.adjustments_applied))


def _multiplier_row(decision_date, adjustment_stack_enabled: bool) -> dict:
    adjusted, unadjusted, set_id, applied = effective_threshold_multiplier(
        decision_date, adjustment_stack_enabled=adjustment_stack_enabled)
    return {
        MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE: adjusted,
        f"{MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE}_unadjusted": unadjusted,
        "adjustment_set_id": set_id,
        "applied_adjustment_ids": applied,
    }


@frame_step(reads=["decision_date", "adjustment_stack_enabled"],
            writes=[MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE, f"{MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE}_unadjusted",
                    "adjustment_set_id", "applied_adjustment_ids"])
def mule_scam_amount_multiplier_step(df: pl.DataFrame) -> pl.DataFrame:
    """A `frame_step`, not a scalar `step`: `applied_adjustment_ids` is a ragged `list[str]` output
    (00 NOTES.md "Framework friction" 4.2: a `frame_step`'s terminal output may not be `list[struct]`,
    but a flat `list[str]` -- this one -- serialises fine). `adjustment_stack_enabled` arrives as a
    column, written upstream by `adjustment_stack_enabled_step` -- see its docstring."""
    rows = [_multiplier_row(d, e) for d, e in zip(df["decision_date"].to_list(),
                                                    df["adjustment_stack_enabled"].to_list())]
    return df.with_columns(pl.DataFrame(rows, schema={
        MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE: pl.Float64, f"{MULE_SCAM_AMOUNT_MULTIPLIER_FEATURE}_unadjusted":
        pl.Float64, "adjustment_set_id": pl.Utf8, "applied_adjustment_ids": pl.List(pl.Utf8)}))


def at_web_action_escalation(family: str, channel_code: int, decision_date: date,
                              adjustment_stack_enabled: bool = True) -> tuple[bool, str]:
    """Whether the account-takeover web-channel escalation is in force and in scope for this event.

    Returns `(escalated, adjustment_id_or_empty)`; `action_resolution.py` uses this to raise a
    firing MS/AT `monitor` action to `step_up` before precedence resolution (§6.5 example).
    """
    if family != vocab.FAMILY_ACCOUNT_TAKEOVER or channel_code != 4:
        return False, ""
    result = OVERLAY_REGISTER.apply_stack(
        "at_web_action", float(vocab.ACTION_MONITOR), {"family": family, "channel_code": channel_code},
        decision_date, ADJUSTMENT_SET_ID, stack_enabled=adjustment_stack_enabled,
    )
    escalated = result.adjusted_value != result.unadjusted_value
    applied = result.adjustments_applied[0] if result.adjustments_applied else ""
    return escalated, applied
