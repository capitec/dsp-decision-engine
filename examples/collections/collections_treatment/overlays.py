"""Overlay stacks that cannot touch a statutory protection, renamed so several share one `dag`."""
from __future__ import annotations

from credit_core.adjustments import AdjustmentStack

# Suspensions, notice periods, contact caps and hours, and prescription belong to Compliance. An overlay may make
# the bank do less than the matrix says, never more than the law allows.
STATUTORY_TARGETS = frozenset({
    "suspension_evaluation", "notice_period_days", "contact_frequency_cap", "permitted_contact_hours",
    "prescription_date", "prescription_handling",
})


def in_dag(stack: AdjustmentStack, applied: str) -> AdjustmentStack:
    """The stack with its fixed output columns renamed after it, so several stacks can share one `dag`.

    `applied` is the column that lists the ids of the overlays that applied. A stack that overlays a statutory
    target (a contact cap, a notice period) raises `ValueError`.
    """
    for a in stack.adjustments:
        if a.target in STATUTORY_TARGETS:
            raise ValueError(f"{a.adjustment_id}: {a.target!r} is a statutory parameter and cannot be overlaid")
    return stack.relabel(writes={
        "adjustments_applied": applied, "adjustment_set_id": f"{stack.name}_set_id",
        stack.unadjusted_output or f"{stack.target}_unadjusted": f"{stack.name}_unadjusted"})
