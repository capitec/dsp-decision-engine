"""Overlays over the published tree: approved, dated changes that leave the tree itself untouched.

Two kinds. A threshold shift (a volume dial, a cut-off shift) moves a named param of the tree, so the tree document and
its node ids stay as published; `resolve` turns the overlays in force on a cycle date into the params document. A cap
reduction scales the tier ceiling the tree answered with: a `CapReduction` document, built on `credit_core.adjustments`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from decider import allow_fallback

from credit_core.adjustments import AdjustmentStack, evaluate

CAMPAIGN_ID = 23


@dataclass(frozen=True)
class TreeOverlay:
    """A threshold-shift overlay of one campaign's tree. `tree_param` names a `{"param": ...}` in the tree document.

    Only floors exist so far (`x >= threshold`), so a shift must raise the threshold: overlays only tighten.
    """

    overlay_id: str
    kind: str  # "volume_dial" or "cutoff_shift"
    tree_param: str
    published_value: float
    new_value: float
    campaign_id: int
    owner: str
    approval_reference: str
    rationale: str
    effective_from: date
    effective_to: date
    review_date: date

    def __post_init__(self) -> None:
        if self.new_value <= self.published_value:
            raise ValueError(f"{self.overlay_id}: {self.tree_param} {self.published_value} -> {self.new_value} loosens a floor")

    def in_force(self, on: date) -> bool:
        return self.effective_from <= on < self.effective_to


TREE_OVERLAYS = (
    TreeOverlay(
        "ADJ-C23-VOL-001", "volume_dial", "n6_discretionary_income_floor", 2200.0, 2600.0, CAMPAIGN_ID,
        "Campaign owner (Flex Loan top-up)", "FORUM-2026-09-14",
        "Bring targeted volume from 412 000 to the 350 000 the contact centre has committed to staff",
        date(2026, 9, 1), date(2026, 12, 1), date(2026, 11, 15)),
    TreeOverlay(
        "ADJ-C23-CUT-001", "cutoff_shift", "n3_behaviour_score_floor", 620.0, 640.0, CAMPAIGN_ID,
        "Credit Risk Policy", "CRC-2026-038", "Early-arrears deterioration on the August cohort",
        date(2026, 10, 1), date(2026, 12, 1), date(2027, 4, 1)),
)


def resolve(overlays: tuple[TreeOverlay, ...], campaign_id: int, cycle_date: date, tree_params: set[str],
            ) -> tuple[dict[str, float], str]:
    """The tree params to run a campaign's cycle with, and the `overlay_stack_id` recording which overlays set them.

    The overlays in force on `cycle_date` (never today, so a re-run of an old cycle resolves its own). An overlay past
    its review date, or naming a param the tree does not have, raises: the cycle must not run under it.

        params, stack_id = resolve(TREE_OVERLAYS, 23, date(2026, 9, 24), tree_params)
        # {"n6_discretionary_income_floor": 2600.0}, "OS-ADJ-C23-VOL-001"
    """
    applied = [o for o in overlays if o.campaign_id == campaign_id and o.in_force(cycle_date)]
    for o in applied:
        if cycle_date >= o.review_date:
            raise ValueError(f"{o.overlay_id} is past its review date {o.review_date} without renewal")
        if o.tree_param not in tree_params:
            raise ValueError(f"{o.overlay_id} shifts {o.tree_param!r}, which campaign {campaign_id}'s tree does not have")
    return ({o.tree_param: o.new_value for o in applied},
            "OS-" + "+".join(o.overlay_id for o in applied) if applied else "OS-NONE")


@allow_fallback
def cap_reduction(
    tier_ceiling: float, campaign_id: int, offer_tier_code: int, cycle_date: date, overlays: tuple, set_id: str,
    adjustment_stack_enabled: bool,
) -> tuple[float, float, str, list[str]]:
    scope = {"campaign_id": campaign_id, "offer_tier_code": offer_tier_code}
    value, applied, _ = evaluate(overlays, tier_ceiling, scope, cycle_date.toordinal(), adjustment_stack_enabled)
    return value, tier_ceiling, set_id, applied


class CapReduction(AdjustmentStack):
    """The cap-reduction document: overlays on the tier ceiling, scoped by `campaign_id` and `offer_tier_code`.

    The param `adjustment_stack_enabled` switches them all off for the shadow run.
    """

    type: Literal["cap_reduction"] = "cap_reduction"

    def to_ir(self, ctx):
        return ctx.call(
            self, cap_reduction, inputs={"tier_ceiling": self.target},
            values={"overlays": self.packed(), "set_id": self.set_id, "adjustment_stack_enabled": self.stack_enabled},
            outputs=("advertised_tier_ceiling", "tier_ceiling_before_cap_overlay", "cap_adjustment_set_id",
                     "cap_adjustments_applied"))
