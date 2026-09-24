"""overlays.py: threshold-shift and cap-reduction overlays (spec 04 §5.3.4)."""
from __future__ import annotations

from datetime import date

import pytest

from campaign_trees.overlays import (
    CAP_ADJUSTMENTS, CUTOFF_SHIFT, TREE_OVERLAYS, TreeOverlay, VOLUME_DIAL, overlay_errors,
    overlay_stack_id, resolve_tree_params,
)
from credit_core.adjustments import Adjustment, AdjustmentEffect


def test_tighten_only_rejects_a_loosening_threshold_shift():
    """§5.3.4 requirement: threshold-shift overlays may only tighten. Raising a `>=` floor
    is tighter; lowering it is a rejected definition, not a silent no-op."""
    with pytest.raises(ValueError, match="loosens"):
        TreeOverlay(
            overlay_id="BAD-001", kind=VOLUME_DIAL, tree_param="x", published_value=2200.0, new_value=1800.0,
            scope={}, owner="x", approval_reference="x", rationale="x",
            effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 6, 1),
        )


def test_cap_reduction_reuses_credit_core_tighten_only_check():
    """The cap-reduction overlay is `credit_core.adjustments.Adjustment` unmodified -- its
    own tighten-only enforcement (a `multiply <= 1.0` for `kind="cap_adjustment"`) rejects a
    loosening cap at definition, for free."""
    with pytest.raises(ValueError, match="loosens"):
        Adjustment(
            adjustment_id="BAD-CAP", kind="cap_adjustment", target="tier_ceiling",
            effect=AdjustmentEffect("multiply", 1.2), scope={}, stack_position=1,
            owner="x", approval_reference="x", rationale="x",
            effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 6, 1), tighten_only=True,
        )


def test_resolve_tree_params_respects_scope_and_effective_window():
    resolved, applied = resolve_tree_params(TREE_OVERLAYS, {"campaign_id": 23}, date(2026, 9, 24))
    assert applied == ("ADJ-C23-VOL-001",)  # the cut-off shift starts 2026-10-01 -- not yet
    assert resolved["campaign23_node6_discretionary_income_thresh"] == 2600.0

    resolved2, applied2 = resolve_tree_params(TREE_OVERLAYS, {"campaign_id": 23}, date(2026, 10, 15))
    assert set(applied2) == {"ADJ-C23-VOL-001", "ADJ-C23-CUT-001"}

    resolved3, applied3 = resolve_tree_params(TREE_OVERLAYS, {"campaign_id": 99}, date(2026, 10, 15))
    assert applied3 == ()  # out of scope -- not an error, correctly not applied


def test_overlay_stack_id_is_order_sensitive_and_stable():
    assert overlay_stack_id(()) == "OS-NONE"
    a = overlay_stack_id(("ADJ-1", "ADJ-2"))
    b = overlay_stack_id(("ADJ-2", "ADJ-1"))
    assert a != b  # composition order matters (§5.3.4 requirement 3)
    assert overlay_stack_id(("ADJ-1", "ADJ-2")) == a  # deterministic


def test_overlay_past_review_date_surfaces():
    stale = TreeOverlay(
        overlay_id="STALE-001", kind=CUTOFF_SHIFT, tree_param="x", published_value=620.0, new_value=640.0,
        scope={"campaign_id": 23}, owner="x", approval_reference="x", rationale="x",
        effective_from=date(2020, 1, 1), effective_to=None, review_date=date(2020, 7, 1),
    )
    errors = overlay_errors([stale], as_of=date(2026, 9, 24))
    assert errors and "STALE-001" in errors[0]


def test_cap_adjustment_register_is_in_force_for_its_declared_cycle():
    result = CAP_ADJUSTMENTS.apply_stack(
        "tier_ceiling", 250_000.0, {"campaign_id": 23, "offer_tier_code": 1}, date(2026, 10, 15),
        "AS-CAMPAIGN23-2026.09",
    )
    assert result.adjusted_value == pytest.approx(200_000.0)
    assert result.unadjusted_value == 250_000.0
