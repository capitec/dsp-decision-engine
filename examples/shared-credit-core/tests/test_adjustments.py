from datetime import date

import pytest

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister

JUNE = date(2026, 6, 1)


def overlay(**changes):
    fields = dict(
        adjustment_id="ADJ-1", kind="score_shift", target="score", effect=AdjustmentEffect("add", -10.0),
        scope={"segment_code": 3}, stack_position=1, owner="Credit Risk Policy", approval_reference="CRC-1",
        rationale="test", effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31))
    return Adjustment(**{**fields, **changes})


def apply(register, provenance=None, on=JUNE, base=600.0, **kwargs):
    return register.apply_stack("score", base, {"segment_code": 3} if provenance is None else provenance, on, "AS-1", **kwargs)


def test_the_unadjusted_value_survives_beside_the_adjusted_one():
    result = apply(AdjustmentRegister([overlay()]))
    assert (result.unadjusted_value, result.adjusted_value) == (600.0, 590.0)


def test_stack_order_is_declared_and_changes_the_answer():
    add, double = dict(adjustment_id="A", effect=AdjustmentEffect("add", -10.0)), dict(
        adjustment_id="B", kind="odds_multiplier", effect=AdjustmentEffect("multiply", 2.0))
    add_first = AdjustmentRegister([overlay(**add, stack_position=1), overlay(**double, stack_position=2)])
    double_first = AdjustmentRegister([overlay(**add, stack_position=2), overlay(**double, stack_position=1)])
    assert apply(add_first, base=100.0).adjusted_value == 180.0     # (100 - 10) * 2
    assert apply(double_first, base=100.0).adjusted_value == 190.0  # 100 * 2 - 10
    assert apply(double_first, base=100.0).adjustments_applied == ("B", "A")


def test_overlays_with_different_scopes_both_apply_to_a_record_in_both():
    register = AdjustmentRegister([
        overlay(adjustment_id="SEG", stack_position=1),
        overlay(adjustment_id="CHAN", stack_position=2, scope={"channel_code": 4}, effect=AdjustmentEffect("add", -5.0))])
    result = apply(register, {"segment_code": 3, "channel_code": 4}, base=665.0)
    assert (result.adjustments_applied, result.adjusted_value) == (("SEG", "CHAN"), 650.0)
    assert apply(register, {"segment_code": 3}, base=665.0).adjustments_applied == ("SEG",)


def test_an_overlay_lapses_at_its_end_date_and_the_lapse_is_recorded():
    register = AdjustmentRegister([overlay(effective_to=date(2026, 6, 1))])
    on_the_last_day = apply(register, on=date(2026, 5, 31))
    assert on_the_last_day.adjustments_applied == ("ADJ-1",)
    lapsed = apply(register, on=date(2026, 6, 1))
    assert lapsed.adjustments_applied == () and lapsed.evaluations[0].reason == "lapsed"


def test_an_overlay_not_yet_effective_is_recorded_as_such():
    result = apply(AdjustmentRegister([overlay()]), on=date(2025, 12, 31))
    assert result.evaluations[0].reason == "not_yet_effective" and result.adjusted_value == 600.0


def test_overlays_past_their_review_date_are_surfaced_until_they_lapse():
    register = AdjustmentRegister([overlay(review_date=date(2026, 3, 1), effective_to=date(2027, 1, 1))])
    assert [a.adjustment_id for a in register.due_for_review(date(2026, 4, 1))] == ["ADJ-1"]
    assert register.due_for_review(date(2026, 2, 28)) == ()
    assert register.due_for_review(date(2027, 1, 1)) == ()


def test_a_stale_overlay_with_no_end_date_is_still_discoverable():
    register = AdjustmentRegister([overlay(effective_from=date(2023, 1, 1), review_date=date(2023, 6, 1))])
    assert [a.adjustment_id for a in register.due_for_review(date(2026, 9, 24))] == ["ADJ-1"]


def test_the_ageing_report_lists_the_oldest_review_date_first():
    register = AdjustmentRegister([overlay(adjustment_id="B", review_date=date(2026, 3, 1)),
                                   overlay(adjustment_id="A", review_date=date(2026, 2, 1))])
    assert [a.adjustment_id for a in register.due_for_review(date(2026, 4, 1))] == ["A", "B"]


def test_applying_one_overlay_outside_its_scope_is_an_error():
    with pytest.raises(ValueError, match="scope"):
        overlay().apply(600.0, {"segment_code": 9}, JUNE)


def test_applying_one_overlay_outside_its_window_is_an_error():
    with pytest.raises(ValueError, match="not in force"):
        overlay().apply(600.0, {"segment_code": 3}, date(2025, 1, 1))


def test_a_stack_skips_out_of_scope_overlays_and_records_why():
    result = apply(AdjustmentRegister([overlay()]), {"segment_code": 9})
    assert result.adjusted_value == 600.0 and result.evaluations[0].reason == "out_of_scope"


def test_every_overlay_considered_is_in_the_trace_whether_or_not_it_applied():
    register = AdjustmentRegister([overlay(adjustment_id="A"), overlay(adjustment_id="B", scope={"segment_code": 1})])
    assert {e.adjustment_id: e.applied for e in apply(register).evaluations} == {"A": True, "B": False}


def test_a_disabled_overlay_never_applies():
    assert apply(AdjustmentRegister([overlay(enabled=False)])).adjusted_value == 600.0


def test_the_stack_can_be_switched_off_through_the_same_call():
    register = AdjustmentRegister([overlay()])
    off = apply(register, stack_enabled=False)
    assert off.adjusted_value == off.unadjusted_value == 600.0
    assert off.evaluations[0].reason == "stack_disabled" and off.adjustments_applied == ()


def test_only_overlays_for_the_target_apply():
    register = AdjustmentRegister([overlay(target="probability_of_default")])
    assert apply(register).adjusted_value == 600.0 and len(register.for_target("probability_of_default")) == 1


@pytest.mark.parametrize("kind,effect", [
    ("score_shift", AdjustmentEffect("add", 10.0)),           # a higher score loosens
    ("odds_multiplier", AdjustmentEffect("multiply", 0.9)),   # lower odds loosen
    ("rate_addon", AdjustmentEffect("add", -0.01)),
    ("cap_adjustment", AdjustmentEffect("multiply", 1.1)),
    ("buffer_adjustment", AdjustmentEffect("add", -0.02)),
])
def test_a_tighten_only_overlay_that_loosens_is_rejected_when_defined(kind, effect):
    with pytest.raises(ValueError, match="tighten-only but its effect loosens"):
        overlay(kind=kind, effect=effect, tighten_only=True)


def test_a_tighten_only_overlay_that_tightens_is_accepted():
    assert overlay(kind="odds_multiplier", effect=AdjustmentEffect("multiply", 1.35), tighten_only=True).tighten_only


def test_tighten_only_needs_a_kind_with_a_known_direction():
    with pytest.raises(ValueError, match="no declared tighten direction"):
        overlay(kind="calibration_reanchor", effect=AdjustmentEffect("add", 1.0), tighten_only=True)


def test_a_loosening_overlay_is_allowed_when_not_declared_tighten_only():
    assert apply(AdjustmentRegister([overlay(effect=AdjustmentEffect("add", 10.0))])).adjusted_value == 610.0


def test_the_set_operation_replaces_the_value():
    assert apply(AdjustmentRegister([overlay(kind="cap_adjustment", effect=AdjustmentEffect("set", 550.0))])).adjusted_value == 550.0


def test_duplicate_overlay_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        AdjustmentRegister([overlay(adjustment_id="A"), overlay(adjustment_id="A")])


def test_the_step_overlays_the_base_column_and_the_stack_flag_switches_it_off(score):
    step = AdjustmentRegister([overlay()]).apply_stack_step("score", "AS-1", base_field="score_raw")
    record = {"score_raw": 600.0, "decision_date": JUNE, "segment_code": 3}
    on = score(step, record)
    off = score(step, record, {"apply_score_adjustments": {"adjustment_stack_enabled": False}})
    assert (on["score"], on["score_unadjusted"], on["adjustments_applied"], on["adjustment_set_id"]) == (590.0, 600.0, ["ADJ-1"], "AS-1")
    assert (off["score"], off["score_unadjusted"], off["adjustments_applied"]) == (600.0, 600.0, [])


def test_the_step_matches_scope_on_the_record_and_skips_overlays_whose_field_is_missing(score):
    step = AdjustmentRegister([overlay()]).apply_stack_step("score", "AS-1", base_field="raw")
    assert score(step, {"raw": 600.0, "decision_date": JUNE})["adjustments_applied"] == []
    assert score(step, {"raw": 600.0, "decision_date": JUNE, "segment_code": 3})["adjustments_applied"] == ["ADJ-1"]
