"""Overlay position 1 (spec 05 §5.5): the event-threshold overlay, factored out of
`pipeline.py` so `counterfactual.py` can reuse the exact same register without
importing the top-level `pipeline.py` entry point (every project's own `pipeline.py`
shares that filename -- see `sole_proprietor.py`'s docstring for why importing another
project's by name is unsafe; the same risk applies to a module importing *its own*
project's top-level entry point from inside its own package, so the register lives
here instead, and `pipeline.py` imports it like everything else does).

"Halve the judgment materiality threshold for sector 412 (construction) for two
quarters" -- spec 05 §5.5's own worked example, reproduced here as a live,
tighten-only overlay.
"""
from __future__ import annotations

from datetime import date

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister

EVENT_ADJUSTMENT_SET_ID = "AS-05-EVT-2026.09"
EVENT_THRESHOLD_OVERLAYS = AdjustmentRegister([
    Adjustment(
        # kind="cap_adjustment" (not "buffer_adjustment"): 00's tighten-direction table
        # (`credit_core/adjustments.py` `_TIGHTEN_RULES`) only recognises `buffer_adjustment`
        # as tightening via an "add" with a non-negative value (a buffer that only grows);
        # a materiality *threshold* tightens the opposite way -- multiplied down, like a cap.
        # `cap_adjustment`'s rule ("multiply by <= 1.0 tightens") is the one that actually
        # matches a threshold ceiling being lowered. See NOTES.md "Framework friction".
        adjustment_id="ADJ-05-014", kind="cap_adjustment", target="material_threshold",
        effect=AdjustmentEffect("multiply", 0.5), scope={"sector_code": 412}, stack_position=1,
        owner="Business Credit Risk Policy", approval_reference="CRC-2026-071",
        rationale="Halve judgment materiality thresholds for construction, two quarters",
        effective_from=date(2026, 1, 1), effective_to=date(2026, 12, 31), review_date=date(2026, 11, 1),
        tighten_only=True,
    ),
])
