"""§5.4: the matrix is 5 376 cells at full declared size, and its overlays never touch
a statutory target."""
from datetime import date

import pytest
from decider import Engine

from collections_treatment import matrix, vocab
from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister


def test_matrix_is_the_full_5376_cells():
    table = matrix.build_treatment_matrix()
    assert len(table.rows.data) == 8 * 6 * 7 * 4 * 4 == 5376


def test_matrix_lookup_returns_a_valid_treatment_and_modifiers():
    table = matrix.build_treatment_matrix()
    out = Engine().bind(table).score({
        "arrears_bucket_code": 8, "collections_band_code": 6, "balance_band_code": 7,
        "contact_band_code": 4, "product_family_code": 4,
    })
    assert out["matrix_treatment_code"] in vocab.TREATMENT_POOL
    assert 1 <= out["matrix_treatment_intensity"] <= 5
    # Bucket 8 (365+ days) must never land on an automated-only treatment (§4.3/§5.4's
    # own escalation shape): pre-legal notice or legal handover, not SMS/email/IVM.
    assert out["matrix_treatment_code"] in (vocab.PRE_LEGAL_NOTICE, vocab.LEGAL_HANDOVER)


def test_intensity_dial_escalates_bucket_3_by_one_level():
    # `adjustment_stack_enabled` is a param() (a knob set through the params document),
    # never a request column -- `.score(record, params={node_name: {...}})`, not a key
    # inside `record` itself. See NOTES.md "Framework friction" for how easy this is to
    # get wrong: passing it inside `record` is silently ignored rather than erroring.
    record = {"matrix_treatment_intensity": 2, "decision_date": date(2026, 8, 15), "arrears_bucket_code": 3}
    out_off = Engine().bind(matrix.intensity_dial_step).score(
        record, params={"_intensity_dial": {"adjustment_stack_enabled": False}},
    )
    out_on = Engine().bind(matrix.intensity_dial_step).score(
        record, params={"_intensity_dial": {"adjustment_stack_enabled": True}},
    )
    assert out_off["treatment_intensity"] == 2
    assert out_on["treatment_intensity"] == 3
    assert out_on["intensity_dial_adjustments_applied"] == ["ADJ-08-MATRIX-001"]


def test_intensity_dial_does_not_apply_outside_its_scoped_bucket():
    out = Engine().bind(matrix.intensity_dial_step).score({
        "matrix_treatment_intensity": 2, "decision_date": date(2026, 8, 15), "arrears_bucket_code": 4,
    })
    assert out["treatment_intensity"] == 2
    assert out["intensity_dial_adjustments_applied"] == []


def test_field_visit_suppression_falls_back_to_no_action():
    out = Engine().bind(matrix.treatment_suppression_step).score({
        "matrix_treatment_code": vocab.FIELD_VISIT, "decision_date": date(2026, 8, 1),
    })
    assert out["treatment_code"] == vocab.NO_ACTION
    assert out["suppression_adjustments_applied"] == ["ADJ-08-MATRIX-002"]


def test_a_suspended_suspension_expiry_or_frequency_cap_can_never_be_named_by_an_overlay():
    bad = AdjustmentRegister([
        Adjustment(
            adjustment_id="ADJ-BAD-001", kind="cap_adjustment", target="contact_frequency_cap",
            effect=AdjustmentEffect("multiply", 2.0), scope={}, stack_position=1,
            owner="nobody", approval_reference="x", rationale="x",
            effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31),
        ),
    ])
    with pytest.raises(ValueError, match="statutory"):
        matrix.assert_no_statutory_target(bad)
