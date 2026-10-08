"""The overlay register: approved, scoped, expiring tightenings of four values, each resolved once per decision."""
from __future__ import annotations

from datetime import date

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister

ADJUSTMENT_SET_ID = "AS-2026.09"


def _overlay(adjustment_id: str, kind: str, target: str, effect: AdjustmentEffect, scope: dict, approval: str,
             rationale: str, effective_to: date) -> Adjustment:
    return Adjustment(
        adjustment_id=adjustment_id, kind=kind, target=target, effect=effect, scope=scope, stack_position=1,
        owner="Credit Risk Policy", approval_reference=approval, rationale=rationale,
        effective_from=date(2026, 1, 1), effective_to=effective_to, review_date=date(2027, 6, 30), tighten_only=True)


OVERLAYS = AdjustmentRegister([
    _overlay("ADJ-0061", "odds_multiplier", "probability_of_default", AdjustmentEffect("multiply", 1.18),
             {"channel_code": 3}, "CC-2027-04", "Channel 3 defaults above model prediction for two quarters",
             date(2027, 9, 30)),
    _overlay("ADJ-0087", "cap_adjustment", "amount_cap", AdjustmentEffect("multiply", 0.90), {"channel_code": 3},
             "CC-2027-11", "Channel 3 amount cap tightened pending a portfolio review", date(2027, 12, 31)),
    _overlay("ADJ-0114", "buffer_adjustment", "affordability_buffer", AdjustmentEffect("add", 0.04),
             {"channel_code": 3}, "CC-2027-07", "Channel 3 buffer raised pending an affordability review",
             date(2027, 12, 31)),
    _overlay("ADJ-0132", "rate_addon", "nominal_annual_rate", AdjustmentEffect("add", 0.006), {"segment_code": 4},
             "CC-2027-13", "60 basis point add-on for segment 4 pending the next card refresh", date(2027, 12, 31)),
])


def overlay_step(target: str, prefix: str, *, base_field: str, adjusted_output: str | None = None,
                 unadjusted_output: str | None = None):
    """The register applied to `base_field`. Its set id and applied list are written as `<prefix>_adjustment_set_id`
    and `<prefix>_adjustments_applied`, because one decision applies the register to several values."""
    return OVERLAYS.apply_stack_step(
        target, ADJUSTMENT_SET_ID, base_field=base_field, adjusted_output=adjusted_output,
        unadjusted_output=unadjusted_output,
    ).relabel(writes={"adjustment_set_id": f"{prefix}_adjustment_set_id",
                      "adjustments_applied": f"{prefix}_adjustments_applied"})
