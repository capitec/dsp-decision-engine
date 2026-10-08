import json
from datetime import date

import polars as pl
import pytest

from limit_mgmt import overlays
from limit_mgmt.matrix import build_matrix_table, generate_matrix_rows, matrix_document
from population import ROOT


def test_the_matrix_has_every_one_of_the_1152_cells_once():
    rows = generate_matrix_rows()
    assert len(rows) == 1_152
    assert len({(r["product"], r["grade"], r["util_band"], r["mob_band"]) for r in rows}) == 1_152


def test_the_matrix_document_on_disk_is_what_the_generator_writes():
    assert json.loads((ROOT / "configs" / "0.1.0" / "matrix.json").read_text()) == matrix_document()


def test_grade_9_and_worse_get_no_increase_in_any_cell():
    worse = [r for r in generate_matrix_rows() if r["grade"] >= 9]
    assert worse and all(r["multiplier"] == 1.0 and r["max_increase"] == 0.0 for r in worse)


def test_a_young_account_is_capped_at_3000_whatever_its_grade():
    young = [r for r in generate_matrix_rows() if r["mob_band"] == 1 and r["grade"] < 9]
    assert all(r["max_increase"] <= 3_000 for r in young)


def test_the_cell_read_depends_on_all_four_keys():
    keys = pl.DataFrame({"product_code": [20, 20, 21], "behaviour_grade": [2, 2, 2], "utilisation_band": [5, 8, 5],
                         "mob_band": [5, 5, 5]})
    out = build_matrix_table().run(keys)
    assert out["cell_id"].to_list() == [
        "limit_matrix@limit-matrix-2026.09#20|2|5|5", "limit_matrix@limit-matrix-2026.09#20|2|8|5",
        "limit_matrix@limit-matrix-2026.09#21|2|5|5"]
    assert out["multiplier"][1] < out["multiplier"][0]  # a fully drawn account is a worse bet than a busy one


def test_the_dial_applies_to_the_excess_over_one_not_to_the_multiplier():
    result = overlays.MATRIX_ADJUSTMENTS.apply_stack("matrix_multiplier_excess", 0.50, {}, date(2026, 9, 24), overlays.SET_ID)
    assert 1.0 + result.adjusted_value == pytest.approx(1.40)  # a 1.50 cell at an 80% dial; not 1.20


def test_a_dial_outside_its_window_does_not_apply():
    result = overlays.MATRIX_ADJUSTMENTS.apply_stack("matrix_multiplier_excess", 0.50, {}, date(2027, 6, 1), overlays.SET_ID)
    assert result.adjusted_value == 0.50 and result.adjustments_applied == ()


def test_switching_the_stack_off_returns_the_unadjusted_value_through_the_same_call():
    result = overlays.MATRIX_ADJUSTMENTS.apply_stack("matrix_multiplier_excess", 0.50, {}, date(2026, 9, 24),
                                                      overlays.SET_ID, stack_enabled=False)
    assert (result.adjusted_value, result.unadjusted_value) == (0.50, 0.50)


def test_an_overlay_past_its_review_date_is_listed_until_it_lapses():
    assert overlays.due_for_review(date(2026, 9, 24)) == []
    assert overlays.due_for_review(date(2026, 10, 20)) == ["ADJ-07-DIAL-2026-03"]
    assert overlays.due_for_review(date(2026, 11, 2)) == []  # the dial has lapsed by then
