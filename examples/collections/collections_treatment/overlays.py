"""Overlay registers that cannot touch a statutory protection."""
from __future__ import annotations

from typing import Sequence

from credit_core.adjustments import Adjustment, AdjustmentRegister

# Suspensions, notice periods, contact caps and hours, and prescription belong to Compliance. An overlay may make
# the bank do less than the matrix says, never more than the law allows.
STATUTORY_TARGETS = frozenset({
    "suspension_evaluation", "notice_period_days", "contact_frequency_cap", "permitted_contact_hours",
    "prescription_date", "prescription_handling",
})


def register(adjustments: Sequence[Adjustment]) -> AdjustmentRegister:
    """An `AdjustmentRegister` that refuses, when defined, an overlay that names a statutory parameter.

    Example: `register([Adjustment(..., target="contact_frequency_cap", ...)])` raises `ValueError`.
    """
    for a in adjustments:
        if a.target in STATUTORY_TARGETS:
            raise ValueError(f"{a.adjustment_id}: {a.target!r} is a statutory parameter and cannot be overlaid")
    return AdjustmentRegister(adjustments)
