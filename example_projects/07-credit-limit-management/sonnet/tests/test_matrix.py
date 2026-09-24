import datetime

from limit_mgmt.matrix import N_GRADES, N_MOB_BANDS, N_UTIL_BANDS, generate_matrix_rows
from limit_mgmt.overlays import ADJUSTMENT_SET_ID, MATRIX_ADJUSTMENTS
from limit_mgmt.vocab import PRODUCT_ACCESS_FACILITY, PRODUCT_EVERYDAY_CARD


def test_matrix_is_the_full_1152_cells():
    """§5.4: "12 x 8 x 6 x 2 = 1 152 cells" -- kept at full declared size (SCOPE.md rule 1)."""
    rows = generate_matrix_rows()
    assert len(rows) == 2 * N_GRADES * N_UTIL_BANDS * N_MOB_BANDS == 1_152
    keys = {(r["product"], r["grade"], r["util_band"], r["mob_band"]) for r in rows}
    assert len(keys) == 1_152  # every cell distinct, no duplicate/missing


def test_multipliers_in_range():
    for row in generate_matrix_rows():
        assert 1.00 <= row["multiplier"] <= 1.75
        assert row["max_increase"] >= 0.0
        assert 500.0 <= row["min_increment"] <= 2_500.0


def test_grade_9_and_worse_gets_no_increase():
    """§5.4: "a grade-9 account gets 1.00 everywhere."""
    for row in generate_matrix_rows():
        if row["grade"] >= 9:
            assert row["multiplier"] == 1.00
            assert row["max_increase"] == 0.0


def test_cycle_dial_applies_to_the_excess_over_one(monkeypatch):
    """§5.4: "a cell multiplier of 1.50 becomes 1.40 at an 80% dial, not 1.20." """
    result = MATRIX_ADJUSTMENTS.apply_stack(
        "matrix_multiplier_excess", 0.50, {}, datetime.date(2026, 9, 24), ADJUSTMENT_SET_ID,
    )
    final_multiplier = 1.0 + result.adjusted_value
    assert final_multiplier == 1.40
    assert final_multiplier != 1.20  # naive multiply-the-whole-value would give this


def test_dial_out_of_effective_window_does_not_apply():
    result = MATRIX_ADJUSTMENTS.apply_stack(
        "matrix_multiplier_excess", 0.50, {}, datetime.date(2027, 6, 1), ADJUSTMENT_SET_ID,
    )
    assert result.adjusted_value == 0.50
    assert result.adjustments_applied == ()
