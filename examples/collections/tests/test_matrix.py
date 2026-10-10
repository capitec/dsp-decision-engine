import copy
from datetime import date

import pytest
from decikit import InterpretedEngine, dag

from collections_treatment import matrix, vocab as v
from collections_treatment.overlays import in_dag

KEYS = ("bucket", "band", "balance_band", "contact_band", "product_family")


def test_the_matrix_is_the_full_5376_cells_each_addressable_once(matrix_document):
    rows = matrix_document["rows"]
    assert len(rows) == 8 * 6 * 7 * 4 * 4 == 5376
    assert len({tuple(row[k] for k in KEYS) for row in rows}) == 5376


def test_the_sample_account_reads_the_cell_its_keys_name(decide, matrix_document):
    out = decide()
    assert out["matrix_cell_id"] == "treatment_matrix@matrix-2026.09#3|3|3|2|1"
    cell = next(r for r in matrix_document["rows"] if (r["bucket"], r["band"], r["balance_band"], r["contact_band"],
                                                         r["product_family"]) == (3, 3, 3, 2, 1))
    assert out["matrix_treatment_code"] == cell["treatment_code"] == v.AUTOMATED_VOICE_MESSAGE
    assert out["matrix_cooling_off_days"] == cell["cooling_off_days"]


def test_the_oldest_arrears_never_get_an_automated_only_treatment(matrix_document):
    codes = {row["treatment_code"] for row in matrix_document["rows"] if row["bucket"] == 8}
    assert codes == {v.PRE_LEGAL_NOTICE, v.LEGAL_HANDOVER}


def test_every_cell_holds_a_known_treatment_and_in_range_modifiers(matrix_document):
    for row in matrix_document["rows"]:
        assert row["treatment_code"] in v.TREATMENT_NAMES
        assert 1 <= row["treatment_intensity"] <= 5 and 0 <= row["permitted_retries"] <= 4
        assert 0 <= row["cooling_off_days"] <= 21


def test_a_new_matrix_document_changes_the_decision_with_no_code_change(matrix_document, build_pipeline, sample, parse, params):
    emails = copy.deepcopy(matrix_document)
    for row in emails["rows"]:
        row["treatment_code"] = v.AUTOMATED_EMAIL
    pipeline = build_pipeline(emails).compile("interpreted")
    out = pipeline.run(parse(sample | {"consent_withdrawn_channels": []}), params)
    assert out["matrix_treatment_code"] == out["treatment_code"] == v.AUTOMATED_EMAIL


def test_the_cell_not_in_the_matrix_falls_back_to_no_action(decide):
    assert decide(product_family_code=9)["matrix_treatment_code"] == v.NO_ACTION


@pytest.fixture
def dial(stacks):
    unit = dag(in_dag(stacks["intensity_dial"], "intensity_dial_adjustments_applied"), matrix.treatment_intensity_step,
               name="dial")
    return lambda on, bucket=3, intensity=2, enabled=True: InterpretedEngine().bind(unit).run(
        {"matrix_treatment_intensity": intensity, "decision_date": on, "arrears_bucket_code": bucket},
        {"dial": {"intensity_dial": {"adjustment_stack_enabled": enabled}}})


def test_the_intensity_dial_escalates_bucket_three_by_one_level_inside_its_window(dial):
    out = dial(date(2026, 8, 15))
    assert out["treatment_intensity"] == 3 and out["intensity_dial_adjustments_applied"] == ["ADJ-COLL-MATRIX-001"]


def test_the_intensity_dial_leaves_other_buckets_and_other_dates_alone(dial):
    assert dial(date(2026, 8, 15), bucket=4)["treatment_intensity"] == 2
    assert dial(date(2026, 9, 24))["treatment_intensity"] == 2  # window ended 12 September
    assert dial(date(2026, 8, 15), enabled=False)["treatment_intensity"] == 2


def test_the_intensity_dial_never_exceeds_the_top_level(dial):
    assert dial(date(2026, 8, 15), intensity=5)["treatment_intensity"] == 5


def test_field_visits_are_suppressed_to_no_action_and_say_so(decide):
    out = decide(days_past_due=150)
    assert out["matrix_treatment_code"] == v.FIELD_VISIT and out["overlaid_treatment_code"] == v.NO_ACTION
    assert out["suppression_adjustments_applied"] == ["ADJ-COLL-MATRIX-002"]
    assert out["non_selection_reason_code"] == v.NS_OVERLAY_SUPPRESSED


def test_a_field_visit_is_recommended_again_once_the_suppression_lapses(decide):
    out = decide(days_past_due=150, decision_date="2026-10-02", days_since_last_treatment=999)
    assert out["overlaid_treatment_code"] == v.FIELD_VISIT and out["suppression_adjustments_applied"] == []
