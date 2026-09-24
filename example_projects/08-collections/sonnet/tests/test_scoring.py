"""§5.3: the four overlay kinds, and "the unadjusted value survives" on every one of them."""
from datetime import date

from decider import Engine

from collections_treatment import scoring


def test_score_shift_applies_only_to_agency_placed_accounts_and_keeps_the_unadjusted_value():
    step = scoring.score_shift_step
    placed = Engine().bind(step).score({
        "collections_score_raw": 620.0, "decision_date": date(2026, 6, 1), "agency_placed_12m": True,
    })
    not_placed = Engine().bind(step).score({
        "collections_score_raw": 620.0, "decision_date": date(2026, 6, 1), "agency_placed_12m": False,
    })
    assert placed["collections_score"] == 620.0 - 22.0
    assert placed["collections_score_unadjusted"] == 620.0
    assert not_placed["collections_score"] == 620.0


def test_odds_multiplier_only_applies_to_scoped_product_family():
    step = scoring.odds_multiplier_step
    scoped = Engine().bind(step).score({
        "roll_probability_before_overlay": 0.20, "decision_date": date(2026, 7, 1), "product_family_code": 1,
    })
    unscoped = Engine().bind(step).score({
        "roll_probability_before_overlay": 0.20, "decision_date": date(2026, 7, 1), "product_family_code": 2,
    })
    assert scoped["roll_probability"] == round(0.20 * 1.28, 10)
    assert unscoped["roll_probability"] == 0.20


def test_scaling_change_multiplies_the_calibration_scale():
    step = scoring.scaling_change_step
    out = Engine().bind(step).score({
        "scale": 60.0, "decision_date": date(2026, 4, 1), "product_family_code": 2,
    })
    assert out["scale"] == 60.0 * (23.0 / 20.0)
    assert out["scaling_change_adjustments_applied"] == ["ADJ-08-SCORE-003"]


def test_band_boundary_shift_lowers_the_probability_the_band_table_sees():
    # `adjustment_stack_enabled` is a param(), passed through `params=`, not the record --
    # see test_matrix.py's note on the same trap.
    step = scoring.band_boundary_shift_step
    record = {"roll_probability": 0.50, "decision_date": date(2026, 6, 1)}
    on = Engine().bind(step).score(record, params={"band_boundary_shift": {"adjustment_stack_enabled": True}})
    off = Engine().bind(step).score(record, params={"band_boundary_shift": {"adjustment_stack_enabled": False}})
    assert on["roll_probability_for_banding"] == 0.50 - 0.03
    assert off["roll_probability_for_banding"] == 0.50
    assert on["band_boundary_shift_adjustments_applied"] == ["ADJ-08-SCORE-004"]
    assert off["band_boundary_shift_adjustments_applied"] == []


def test_stack_disabled_reproduces_the_unadjusted_base_score_run():
    """§7.6 / acceptance §10 item 8: the base score's own performance is monitored with
    the overlay stack off, through the *same* code path -- one flag, not a fork."""
    step = scoring.score_shift_step
    out = Engine().bind(step).score(
        {"collections_score_raw": 620.0, "decision_date": date(2026, 6, 1), "agency_placed_12m": True},
        params={"_score_shift": {"adjustment_stack_enabled": False}},
    )
    assert out["collections_score"] == out["collections_score_unadjusted"] == 620.0
    assert out["score_shift_adjustments_applied"] == []
