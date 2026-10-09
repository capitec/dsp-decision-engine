import json
from datetime import date

import pytest
from decider import ConfigurableStep, flow
from decider.lab import diff

from credit_core.adjustments import AdjustmentStack, Effect, Overlay

JUNE = date(2026, 6, 1)


def overlay(**changes):
    fields = dict(
        adjustment_id="ADJ-1", kind="score_shift", target="score", effect=Effect(op="add", value=-10.0),
        scope={"segment_code": 3}, stack_position=1, owner="Credit Risk Policy", approval_reference="CRC-1",
        rationale="test", effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31))
    return Overlay(**{**fields, **changes})


def stack(*overlays, **fields):
    return AdjustmentStack(target="score", set_id="AS-1", base_field="score_raw", adjustments=overlays, **fields)


def apply(stack, provenance=None, on=JUNE, base=600.0, **kwargs):
    return stack.apply(base, {"segment_code": 3} if provenance is None else provenance, on, **kwargs)


def same(a, b):
    return diff.structure(flow(a, name="t"), flow(b, name="t")).identical


def edited(a, index=0, **changes):
    return stack(*(o.model_copy(update=changes) if i == index else o for i, o in enumerate(a.adjustments)))


def test_the_unadjusted_value_survives_beside_the_adjusted_one():
    result = apply(stack(overlay()))
    assert (result.unadjusted_value, result.adjusted_value, result.adjustment_set_id) == (600.0, 590.0, "AS-1")


def test_stack_order_is_declared_and_changes_the_answer():
    add = dict(adjustment_id="A", effect=Effect(op="add", value=-10.0))
    double = dict(adjustment_id="B", kind="odds_multiplier", effect=Effect(op="multiply", value=2.0))
    add_first = stack(overlay(**add, stack_position=1), overlay(**double, stack_position=2))
    double_first = stack(overlay(**add, stack_position=2), overlay(**double, stack_position=1))
    assert apply(add_first, base=100.0).adjusted_value == 180.0     # (100 - 10) * 2
    assert apply(double_first, base=100.0).adjusted_value == 190.0  # 100 * 2 - 10
    assert apply(double_first, base=100.0).adjustments_applied == ("B", "A")


def test_overlays_with_different_scopes_both_apply_to_a_record_in_both():
    both = stack(overlay(adjustment_id="SEG"),
                 overlay(adjustment_id="CHAN", stack_position=2, scope={"channel_code": 4}, effect=Effect(op="add", value=-5.0)))
    result = apply(both, {"segment_code": 3, "channel_code": 4}, base=665.0)
    assert (result.adjustments_applied, result.adjusted_value) == (("SEG", "CHAN"), 650.0)
    assert apply(both, {"segment_code": 3}, base=665.0).adjustments_applied == ("SEG",)


def test_an_overlay_lapses_at_its_end_date_and_the_lapse_is_recorded():
    lapsing = stack(overlay(effective_to=date(2026, 6, 1)))
    assert apply(lapsing, on=date(2026, 5, 31)).adjustments_applied == ("ADJ-1",)
    lapsed = apply(lapsing, on=date(2026, 6, 1))
    assert lapsed.adjustments_applied == () and lapsed.evaluations[0].reason == "lapsed"


def test_an_overlay_not_yet_effective_is_recorded_as_such():
    result = apply(stack(overlay()), on=date(2025, 12, 31))
    assert result.evaluations[0].reason == "not_yet_effective" and result.adjusted_value == 600.0


def test_overlays_past_their_review_date_are_surfaced_until_they_lapse():
    ageing = stack(overlay(review_date=date(2026, 3, 1), effective_to=date(2027, 1, 1)))
    assert [a.adjustment_id for a in ageing.due_for_review(date(2026, 4, 1))] == ["ADJ-1"]
    assert ageing.due_for_review(date(2026, 2, 28)) == ()
    assert ageing.due_for_review(date(2027, 1, 1)) == ()


def test_a_stale_overlay_with_no_end_date_is_still_discoverable():
    stale = stack(overlay(effective_from=date(2023, 1, 1), review_date=date(2023, 6, 1)))
    assert [a.adjustment_id for a in stale.due_for_review(date(2026, 9, 24))] == ["ADJ-1"]


def test_the_ageing_report_lists_the_oldest_review_date_first():
    ageing = stack(overlay(adjustment_id="B", review_date=date(2026, 3, 1)),
                   overlay(adjustment_id="A", review_date=date(2026, 2, 1)))
    assert [a.adjustment_id for a in ageing.due_for_review(date(2026, 4, 1))] == ["A", "B"]


def test_a_stack_skips_out_of_scope_overlays_and_records_why():
    result = apply(stack(overlay()), {"segment_code": 9})
    assert result.adjusted_value == 600.0 and result.evaluations[0].reason == "out_of_scope"


def test_every_overlay_considered_is_in_the_trace_whether_or_not_it_applied():
    both = stack(overlay(adjustment_id="A"), overlay(adjustment_id="B", scope={"segment_code": 1}))
    assert {e.adjustment_id: e.applied for e in apply(both).evaluations} == {"A": True, "B": False}


def test_a_disabled_overlay_never_applies():
    assert apply(stack(overlay(enabled=False))).adjusted_value == 600.0


def test_the_stack_can_be_switched_off_through_the_same_call():
    off = apply(stack(overlay()), stack_enabled=False)
    assert off.adjusted_value == off.unadjusted_value == 600.0
    assert off.evaluations[0].reason == "stack_disabled" and off.adjustments_applied == ()


def test_overlays_for_another_target_are_rejected_when_the_document_loads():
    with pytest.raises(ValueError, match="not for target 'score'"):
        stack(overlay(target="probability_of_default"))


@pytest.mark.parametrize("kind,effect", [
    ("score_shift", Effect(op="add", value=10.0)),           # a higher score loosens
    ("odds_multiplier", Effect(op="multiply", value=0.9)),   # lower odds loosen
    ("rate_addon", Effect(op="add", value=-0.01)),
    ("cap_adjustment", Effect(op="multiply", value=1.1)),
    ("buffer_adjustment", Effect(op="add", value=-0.02)),
])
def test_a_tighten_only_overlay_that_loosens_is_rejected_when_the_document_loads(kind, effect):
    with pytest.raises(ValueError, match="tighten-only but its effect loosens"):
        overlay(kind=kind, effect=effect, tighten_only=True)


def test_a_tighten_only_overlay_that_tightens_is_accepted():
    assert overlay(kind="odds_multiplier", effect=Effect(op="multiply", value=1.35), tighten_only=True).tighten_only


def test_tighten_only_needs_a_kind_with_a_known_direction():
    with pytest.raises(ValueError, match="no declared tighten direction"):
        overlay(kind="calibration_reanchor", effect=Effect(op="add", value=1.0), tighten_only=True)


def test_a_loosening_overlay_is_allowed_when_not_declared_tighten_only():
    assert apply(stack(overlay(effect=Effect(op="add", value=10.0)))).adjusted_value == 610.0


def test_the_set_operation_replaces_the_value():
    assert apply(stack(overlay(kind="cap_adjustment", effect=Effect(op="set", value=550.0)))).adjusted_value == 550.0


def test_duplicate_overlay_ids_are_rejected_when_the_document_loads():
    with pytest.raises(ValueError, match="duplicate"):
        stack(overlay(adjustment_id="A"), overlay(adjustment_id="A"))


def test_a_scope_field_the_step_does_not_match_on_is_rejected_when_the_step_is_built():
    sectoral = stack(overlay(scope={"sector_code": 412}))
    with pytest.raises(ValueError, match="sector_code"):
        flow(sectoral, name="t").compile("interpreted")


def test_apply_matches_any_provenance_field():
    sectoral = stack(overlay(scope={"sector_code": 412}))
    assert apply(sectoral, {"sector_code": 412}).adjusted_value == 590.0
    assert apply(sectoral, {"sector_code": 413}).adjusted_value == 600.0


def test_the_step_overlays_the_base_column_and_the_stack_flag_switches_it_off(score):
    step = stack(overlay())
    assert step.name == "apply_score_adjustments"
    record = {"score_raw": 600.0, "decision_date": JUNE, "segment_code": 3}
    on = score(step, record)
    off = score(step, record, {"apply_score_adjustments": {"adjustment_stack_enabled": False}})
    assert (on["score"], on["score_unadjusted"], on["adjustments_applied"], on["adjustment_set_id"]) == (590.0, 600.0, ["ADJ-1"], "AS-1")
    assert (off["score"], off["score_unadjusted"], off["adjustments_applied"]) == (600.0, 600.0, [])


def test_the_step_matches_scope_on_the_record_and_skips_overlays_whose_field_is_missing(score):
    step = stack(overlay())
    assert score(step, {"score_raw": 600.0, "decision_date": JUNE})["adjustments_applied"] == []
    assert score(step, {"score_raw": 600.0, "decision_date": JUNE, "segment_code": 3})["adjustments_applied"] == ["ADJ-1"]


def test_the_stack_flag_is_a_param_with_the_documented_path_and_default():
    assert flow(stack(overlay()), name="p").parameters().defaults() == {"p": {"apply_score_adjustments": {"adjustment_stack_enabled": True}}}


def test_the_document_loads_with_dates_as_json_strings():
    doc = json.loads(stack(overlay(effective_to=date(2027, 1, 1))).model_dump_json())
    assert doc["adjustments"][0]["effective_from"] == "2026-01-01"
    assert ConfigurableStep.load(doc) == stack(overlay(effective_to=date(2027, 1, 1)))


def test_the_same_overlays_are_unchanged_in_the_diff():
    assert same(stack(overlay()), ConfigurableStep.load(stack(overlay()).model_dump_json()))


@pytest.mark.parametrize("change", [
    dict(effect=Effect(op="add", value=-36.0)),                 # the shift doubled
    dict(effect=Effect(op="multiply", value=-10.0)),
    dict(scope={"segment_code": 4}),
    dict(effective_to=date(2027, 1, 1)),
    dict(effective_from=date(2026, 2, 1)),
    dict(enabled=False),
    dict(adjustment_id="ADJ-1B"),
])
def test_an_edited_overlay_shows_in_the_diff(change):
    assert not same(stack(overlay()), edited(stack(overlay()), **change))


def test_swapping_the_stack_order_shows_in_the_diff():
    a, b = overlay(adjustment_id="A"), overlay(adjustment_id="B", effect=Effect(op="add", value=-5.0), stack_position=2)
    swapped = stack(a.model_copy(update={"stack_position": 2}), b.model_copy(update={"stack_position": 1}))
    assert not same(stack(a, b), swapped)


def test_a_governance_only_edit_is_not_behaviour():
    governed = edited(stack(overlay()), owner="Model Risk", approval_reference="CRC-9", rationale="reworded",
                      review_date=date(2027, 3, 31))
    assert same(stack(overlay()), governed)


def test_the_set_id_and_the_suppress_switch_show_in_the_diff():
    assert not same(stack(overlay()), AdjustmentStack(target="score", set_id="AS-2", base_field="score_raw",
                                                       adjustments=[overlay()]))
    assert not same(stack(overlay()), stack(overlay(), stack_enabled={"param": "adjustment_stack_enabled", "default": False}))


def test_the_step_compiles_with_its_python_boundary_declared():
    fused = flow(stack(overlay()), name="t").compile("fused")
    assert all(reason.startswith("@allow_fallback") for reason in fused.fallbacks().values())
