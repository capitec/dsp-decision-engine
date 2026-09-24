"""The one overlay register for this project's slice (spec 10 §5.9's O-05, O-21).

**Resolved once, for the whole decision, by P08** -- and then read by four
later phases that do not own it (P09's cap reduction, P10's buffer
adjustment, P12's rate add-on), which is exactly the composition cost 10
§5.9's "What changes because it is inside this flow" describes: "P08
carries an obligation it does not own and does not benefit from." One
`AdjustmentRegister` instance, defined in one module, imported by every
phase that reads it -- not one register per phase, which would let two
phases resolve the register independently and violate O-05/O-21.

Reused directly from `credit_core.adjustments`: `Adjustment`,
`AdjustmentEffect`, `AdjustmentRegister` -- the whole mechanism. Written
here: this project's own four live overlays, one worked example per kind
this project's spec sections cite (10 §5.9, §5.11, §5.13).
"""
from __future__ import annotations

from datetime import date

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister

ADJUSTMENT_SET_ID = "P10-AS-2026.09"

OVERLAY_REGISTER = AdjustmentRegister([
    # P08 (10 §5.9's worked example): a channel-3 odds multiplier on PD.
    Adjustment(
        adjustment_id="ADJ-0061", kind="odds_multiplier", target="probability_of_default",
        effect=AdjustmentEffect("multiply", 1.18), scope={"channel_code": 3}, stack_position=1,
        owner="Credit Risk Policy", approval_reference="CC-2027-04",
        rationale="Channel 3 odds multiplier: realised default rate above model prediction for two quarters",
        effective_from=date(2026, 1, 1), effective_to=date(2027, 9, 30), review_date=date(2027, 6, 30),
        tighten_only=True,
    ),
    # P09 (10 §5.9's worked example): a policy overlay reducing amount_cap by 10% on channel 3,
    # applied *after* the register's own chain -- "last moved by an overlay" (10 §5.9).
    Adjustment(
        adjustment_id="ADJ-0087", kind="cap_adjustment", target="amount_cap",
        effect=AdjustmentEffect("multiply", 0.90), scope={"channel_code": 3}, stack_position=1,
        owner="Credit Risk Policy", approval_reference="CC-2027-11",
        rationale="Channel 3 amount cap tightened pending a portfolio review",
        effective_from=date(2026, 6, 1), effective_to=date(2027, 12, 31), review_date=date(2027, 3, 31),
        tighten_only=True,
    ),
    # P10 (10 §5.11's worked example): +4pp buffer on channel 3 for the quarter.
    Adjustment(
        adjustment_id="ADJ-0114", kind="buffer_adjustment", target="affordability_buffer",
        effect=AdjustmentEffect("add", 0.04), scope={"channel_code": 3}, stack_position=1,
        owner="Credit Risk Policy", approval_reference="CC-2027-07",
        rationale="Channel 3 buffer raised for the quarter pending an affordability review",
        effective_from=date(2026, 1, 1), effective_to=date(2027, 12, 31), review_date=date(2027, 6, 30),
        tighten_only=True,
    ),
    # P12 (10 §5.13(e)): a rate add-on, basis points, over a grade range, entered without
    # reissuing the whole card. Declared and never merged into the card (10 §5.13(e)).
    Adjustment(
        adjustment_id="ADJ-0132", kind="rate_addon", target="nominal_annual_rate",
        effect=AdjustmentEffect("add", 0.006), scope={"segment_code": 4}, stack_position=1,
        owner="Treasury / Pricing", approval_reference="CC-2027-13",
        rationale="60 basis point add-on for segment 4 pending the next card refresh",
        effective_from=date(2026, 1, 1), effective_to=date(2027, 12, 31), review_date=date(2027, 6, 30),
        tighten_only=True,
    ),
])
