from datetime import date

from business_nested import events, vocab
from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.adverse_events import JUDGMENT, MATERIAL, MINOR


def test_overlaid_threshold_reclassifies_and_records_both_values():
    """Spec 05 §5.5: base and overlaid threshold both recorded, overlay id attached."""
    overlays = AdjustmentRegister([
        Adjustment(
            "ADJ-05-014", "cap_adjustment", "material_threshold", AdjustmentEffect("multiply", 0.5),
            {"sector_code": 412}, 1, "Credit Committee", "CRC-2026-011", "halve construction materiality",
            date(2026, 1, 1), date(2026, 8, 31), date(2026, 7, 1), tighten_only=True,
        ),
    ])
    r = events.classify_events(
        ev_entity_id=[7], ev_event_id=[41], ev_event_type_code=[JUDGMENT], ev_amount=[40000.0],
        ev_status_is_active=[True], ev_is_disputed=[False], ev_is_satisfied=[False],
        entity_id=[7], entity_criticality_class=[vocab.CRITICAL], decision_date=date(2026, 7, 15),
        threshold_overlays=overlays, adjustment_set_id="AS-05-2026.09", stack_enabled=True,
        business_provenance={"sector_code": 412},
    )
    assert r["ev_severity_code"][0] == MATERIAL
    assert r["ev_threshold_used"][0] == 5000.0   # 10000 halved
    assert r["ev_threshold_base"][0] == 10000.0
    assert r["ev_overlay_ids"][0] == "ADJ-05-014"

    # Stack off: same event, unoverlaid thresholds (10000/50000) -- still MATERIAL
    # (40000 is above the base material threshold too), but against the *unoverlaid*
    # base value, not the halved one -- the "stack off" run must be the same call,
    # only the flag differs (spec 00 §7.6).
    stack_off = events.classify_events(
        ev_entity_id=[7], ev_event_id=[41], ev_event_type_code=[JUDGMENT], ev_amount=[40000.0],
        ev_status_is_active=[True], ev_is_disputed=[False], ev_is_satisfied=[False],
        entity_id=[7], entity_criticality_class=[vocab.CRITICAL], decision_date=date(2026, 7, 15),
        threshold_overlays=overlays, adjustment_set_id="AS-05-2026.09", stack_enabled=False,
    )
    assert stack_off["ev_severity_code"][0] == MATERIAL
    assert stack_off["ev_threshold_used"][0] == 10000.0
    assert stack_off["ev_overlay_ids"][0] == ""


def test_disputed_event_downgrades_one_class_and_is_flagged_provisional():
    """AE-C-22 (spec 05 §5.5)."""
    r = events.classify_events(
        ev_entity_id=[1], ev_event_id=[1], ev_event_type_code=[JUDGMENT], ev_amount=[90000.0],
        ev_status_is_active=[True], ev_is_disputed=[True], ev_is_satisfied=[False],
        entity_id=[1], entity_criticality_class=[vocab.CRITICAL], decision_date=date(2026, 1, 1),
        threshold_overlays=AdjustmentRegister([]), adjustment_set_id="AS-none", stack_enabled=True,
    )
    # 90000 > disqualifying(50000) -> would be DISQUALIFYING; disputed downgrades one class.
    assert r["ev_severity_code"][0] == MATERIAL
    assert r["ev_classification_provisional"][0] is True
