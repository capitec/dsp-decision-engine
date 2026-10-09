"""Overlay stacks that cannot touch a statutory protection and may scope on any one column.

`credit_core.adjustments.AdjustmentStack` matches scope on four fixed provenance columns; a collections overlay scopes
on the arrears bucket, the product family or a flag, so this stack reads whichever single column its overlays name.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import model_validator

from decider import allow_fallback

from credit_core.adjustments import AdjustmentStack, evaluate

# Suspensions, notice periods, contact caps and hours, and prescription belong to Compliance. An overlay may make
# the bank do less than the matrix says, never more than the law allows.
STATUTORY_TARGETS = frozenset({
    "suspension_evaluation", "notice_period_days", "contact_frequency_cap", "permitted_contact_hours",
    "prescription_date", "prescription_handling",
})


@allow_fallback
def apply_scoped(
    adjustment_base_value: float, decision_date: date, overlays: tuple, set_id: str, scope_column: str,
    adjustment_stack_enabled: bool, scope_value: float | None = None,
) -> tuple[float, float, str, list[str]]:
    provenance = {} if scope_value is None else {scope_column: scope_value}
    value, applied, _ = evaluate(overlays, adjustment_base_value, provenance, decision_date.toordinal(),
                                 adjustment_stack_enabled)
    return value, adjustment_base_value, set_id, applied


class OverlayStack(AdjustmentStack):
    """An `AdjustmentStack` whose overlays scope on one column of any name, and which refuses statutory targets.

    Example: a stack whose overlays are `scope: {"arrears_bucket_code": 3}` reads the `arrears_bucket_code` column;
    one with an overlay for `contact_frequency_cap` fails to load with `ValueError`.
    """

    type: Literal["collections_overlay_stack"] = "collections_overlay_stack"

    @model_validator(mode="after")
    def _one_scope_column_and_no_statutory_target(self) -> OverlayStack:
        for a in self.adjustments:
            if a.target in STATUTORY_TARGETS:
                raise ValueError(f"{a.adjustment_id}: {a.target!r} is a statutory parameter and cannot be overlaid")
        columns = {k for a in self.adjustments for k in a.scope}
        if len(columns) > 1:
            raise ValueError(f"{self.name}: overlays scope on {sorted(columns)}; a stack reads one scope column")
        return self

    def to_ir(self, ctx: Any) -> Any:
        column = next((k for a in self.adjustments for k in a.scope), "")
        return ctx.call(
            self, apply_scoped,
            inputs={"adjustment_base_value": self.base_field or self.target, **({"scope_value": column} if column else {})},
            values={"overlays": self.packed(), "set_id": self.set_id, "scope_column": column,
                    "adjustment_stack_enabled": self.stack_enabled, **({} if column else {"scope_value": None})},
            outputs=(self.adjusted_output or self.target, self.unadjusted_output or f"{self.target}_unadjusted",
                     "adjustment_set_id", "adjustments_applied"))


def in_dag(stack: OverlayStack, applied: str) -> OverlayStack:
    """The stack with its fixed output columns renamed after it, so several stacks can share one `dag`.

    `applied` is the column that lists the ids of the overlays that applied.
    """
    return stack.relabel(writes={
        "adjustments_applied": applied, "adjustment_set_id": f"{stack.name}_set_id",
        stack.unadjusted_output or f"{stack.target}_unadjusted": f"{stack.name}_unadjusted"})
