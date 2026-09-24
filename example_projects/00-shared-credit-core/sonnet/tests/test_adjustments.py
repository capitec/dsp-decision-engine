"""`core.adjustments`: the six properties of spec 00 §6.22, and addendum B2."""
from datetime import date

import pytest
from decider import Engine

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister


def _adj(**overrides):
    defaults = dict(
        adjustment_id="ADJ-1", kind="score_shift", target="score",
        effect=AdjustmentEffect("add", -10.0), scope={"segment_code": 3}, stack_position=1,
        owner="Credit Risk Policy", approval_reference="CRC-1", rationale="test",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31),
    )
    defaults.update(overrides)
    return Adjustment(**defaults)


# --- property 1/2: overlay, not edit; unadjusted value survives ------------------------------

def test_unadjusted_value_survives_alongside_the_adjusted_one():
    register = AdjustmentRegister([_adj()])
    result = register.apply_stack("score", 600.0, {"segment_code": 3}, date(2026, 6, 1), "AS-1")
    assert result.unadjusted_value == 600.0
    assert result.adjusted_value == 590.0


# --- property 3: order is declared, not emergent, and changes the answer ---------------------

def test_stack_order_is_declared_and_changes_the_answer():
    add_then_double = AdjustmentRegister([
        _adj(adjustment_id="A", stack_position=1, effect=AdjustmentEffect("add", -10.0), tighten_only=False),
        _adj(adjustment_id="B", stack_position=2, kind="odds_multiplier", effect=AdjustmentEffect("multiply", 2.0)),
    ])
    double_then_add = AdjustmentRegister([
        _adj(adjustment_id="A", stack_position=2, effect=AdjustmentEffect("add", -10.0), tighten_only=False),
        _adj(adjustment_id="B", stack_position=1, kind="odds_multiplier", effect=AdjustmentEffect("multiply", 2.0)),
    ])
    r1 = add_then_double.apply_stack("score", 100.0, {"segment_code": 3}, date(2026, 6, 1), "AS-1")
    r2 = double_then_add.apply_stack("score", 100.0, {"segment_code": 3}, date(2026, 6, 1), "AS-1")
    assert r1.adjusted_value == 180.0  # (100-10)*2
    assert r2.adjusted_value == 190.0  # (100*2)-10
    assert r1.adjusted_value != r2.adjusted_value


def test_change_scenario_13_two_adjustments_collide_and_both_apply_in_order():
    """SCOPE.md change scenario 13: a segment shift and a channel multiplier both fire."""
    register = AdjustmentRegister([
        _adj(adjustment_id="SEG", stack_position=1, scope={"segment_code": 3}, effect=AdjustmentEffect("add", -18.0)),
        _adj(adjustment_id="CHAN", stack_position=2, scope={"channel_code": 4},
             kind="rate_addon", effect=AdjustmentEffect("add", -5.0)),
    ])
    result = register.apply_stack("score", 665.0, {"segment_code": 3, "channel_code": 4}, date(2026, 6, 1), "AS-1")
    assert result.adjustments_applied == ("SEG", "CHAN")
    assert result.adjusted_value == 642.0


# --- property 4: identity and justification are mandatory fields -----------------------------

def test_identity_fields_are_present():
    a = _adj()
    assert (a.owner, a.approval_reference, a.rationale) == ("Credit Risk Policy", "CRC-1", "test")


# --- property 5: expiry lapses automatically, and review dates surface -----------------------

def test_an_expired_overlay_lapses_and_the_lapse_is_recorded():
    register = AdjustmentRegister([_adj(effective_to=date(2026, 6, 1))])
    result = register.apply_stack("score", 600.0, {"segment_code": 3}, date(2026, 7, 1), "AS-1")
    assert result.adjustments_applied == ()
    assert result.evaluations[0].reason == "lapsed"


def test_due_for_review_surfaces_overlays_past_their_review_date():
    register = AdjustmentRegister([_adj(review_date=date(2026, 3, 1), effective_to=date(2027, 1, 1))])
    assert [a.adjustment_id for a in register.due_for_review(date(2026, 4, 1))] == ["ADJ-1"]
    assert register.due_for_review(date(2026, 2, 1)) == ()


def test_change_scenario_12_a_stale_tightening_is_discoverable():
    """SCOPE.md scenario 12: a three-year-old tightening still in force must be listable."""
    register = AdjustmentRegister([_adj(
        effective_from=date(2023, 1, 1), effective_to=None, review_date=date(2023, 6, 1),
    )])
    overdue = register.due_for_review(date(2026, 9, 24))
    assert len(overdue) == 1 and overdue[0].adjustment_id == "ADJ-1"


# --- property 6: scope is declared; applying outside it is an error, not a silent no-op ------

def test_applying_a_single_adjustment_outside_its_scope_is_an_error():
    with pytest.raises(ValueError, match="scope"):
        _adj().apply(600.0, {"segment_code": 9}, date(2026, 6, 1))


def test_apply_stack_silently_skips_out_of_scope_overlays_but_records_why():
    register = AdjustmentRegister([_adj()])
    result = register.apply_stack("score", 600.0, {"segment_code": 9}, date(2026, 6, 1), "AS-1")
    assert result.adjusted_value == 600.0
    assert result.evaluations[0].reason == "out_of_scope"


# --- tighten-only, rejected at definition (addendum B2) --------------------------------------

def test_a_loosening_effect_is_rejected_for_a_tighten_only_kind():
    with pytest.raises(ValueError, match="tighten-only"):
        _adj(kind="score_shift", effect=AdjustmentEffect("add", 10.0), tighten_only=True)


def test_a_tightening_effect_is_accepted_for_a_tighten_only_kind():
    a = _adj(kind="odds_multiplier", effect=AdjustmentEffect("multiply", 1.35), tighten_only=True)
    assert a.tighten_only


def test_tighten_only_on_an_undeclared_kind_is_rejected():
    with pytest.raises(ValueError, match="no declared tighten direction"):
        _adj(kind="calibration_reanchor", effect=AdjustmentEffect("add", 1.0), tighten_only=True)


# --- 09 §5.15 item 14: evaluation recorded, not only firing ----------------------------------

def test_every_considered_adjustment_is_in_the_evaluation_trace_even_if_not_applied():
    register = AdjustmentRegister([_adj(adjustment_id="A"), _adj(adjustment_id="B", scope={"segment_code": 1})])
    result = register.apply_stack("score", 600.0, {"segment_code": 3}, date(2026, 6, 1), "AS-1")
    ids = {e.adjustment_id: e.applied for e in result.evaluations}
    assert ids == {"A": True, "B": False}


# --- §7.6 / acceptance §10 item 8: stack-off run, same implementation ------------------------

def test_stack_disabled_returns_the_unadjusted_answer_through_the_same_call():
    register = AdjustmentRegister([_adj()])
    on = register.apply_stack("score", 600.0, {"segment_code": 3}, date(2026, 6, 1), "AS-1", stack_enabled=True)
    off = register.apply_stack("score", 600.0, {"segment_code": 3}, date(2026, 6, 1), "AS-1", stack_enabled=False)
    assert on.adjusted_value == 590.0
    assert off.adjusted_value == off.unadjusted_value == 600.0


def test_register_rejects_duplicate_adjustment_ids():
    with pytest.raises(ValueError, match="duplicate"):
        AdjustmentRegister([_adj(adjustment_id="A"), _adj(adjustment_id="A")])


# --- apply_stack_step: the pipeline-wiring convenience ----------------------------------------

def test_apply_stack_step_wires_into_a_step_and_supports_the_stack_flag():
    register = AdjustmentRegister([_adj()])
    s = register.apply_stack_step("score", "AS-1", base_field="score_raw")
    exe = Engine().bind(s, mode="interpreted")
    record = {"score_raw": 600.0, "decision_date": date(2026, 6, 1), "segment_code": 3}
    on = exe.score(record)
    off = exe.score(record, params={"apply_score_adjustments": {"adjustment_stack_enabled": False}})
    assert on["score"] == 590.0
    assert off["score"] == 600.0
    assert on["score_unadjusted"] == off["score_unadjusted"] == 600.0
    assert on["adjustments_applied"] == ["ADJ-1"]
