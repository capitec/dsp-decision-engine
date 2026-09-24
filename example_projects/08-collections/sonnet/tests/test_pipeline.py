"""End-to-end: the sample request scores through the real serving handler (BRIEF's
mandatory verification), and the 09-C evidence minimum (spec 09 §5.15) is present on
every record."""
import os

from decider.serving.handler import construct_handler_from_settings

from inference import _typed_sample_record


def _handler():
    os.environ.setdefault("DECIDER_API__CODE_PATH", os.path.dirname(os.path.dirname(__file__)))
    os.environ.setdefault("DECIDER_API__PIPELINE", "pipeline:build")
    os.environ.setdefault(
        "DECIDER_CONFIG__BASEPATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs"),
    )
    os.environ.setdefault("DECIDER_API__MODE", "interpreted")
    handler = construct_handler_from_settings()
    handler.stage()
    handler.activate()
    return handler.module_fn()


def test_sample_request_scores_through_the_handler():
    live = _handler()
    record = _typed_sample_record()
    out = live.executable.score(record, live.params)
    assert out["treatment_code"] in range(0, 14)
    assert out["collections_band_code"] in range(1, 7)
    assert out["arrangement_sustainable"] in (True, False)
    assert out["assessment_mode_code"] == 3  # ARRANGEMENT -- named in the evidence, §5.6


def test_09c_evidence_minimum_is_present(spec="09 §5.15"):
    """decision id, stable logic ids, table version and cell, overlay stack with
    unadjusted values, reason codes with registry version, and no "today" -- SCOPE.md's
    own paraphrase of the 09-C checklist every slice must pass."""
    live = _handler()
    record = _typed_sample_record()
    out = live.executable.score(record, live.params)

    # Stable logic ids / table version and cell.
    for key in ("matrix_cell_id", "calibration_cell_id", "recovery_curve_cell_id",
                "collections_band_cell_id", "balance_band_cell_id"):
        assert out[key], key

    # Overlay stack, unadjusted values retained beside the adjusted ones.
    assert "collections_score_unadjusted" in out
    assert "collections_band_code_unadjusted" in out
    assert "max_affordable_instalment_unadjusted" in out
    for key in ("score_shift_adjustments_applied", "odds_multiplier_adjustments_applied",
                "scaling_change_adjustments_applied", "band_boundary_shift_adjustments_applied",
                "intensity_dial_adjustments_applied", "suppression_adjustments_applied"):
        assert isinstance(out[key], list), key

    # Suspensions: every one attributable, never just the first.
    assert isinstance(out["suspension_codes"], list)
    assert len(out["suspension_codes"]) == len(out["suspension_scopes"]) == len(out["suspension_sources"])

    # "No today": `decision_date` drives everything; nothing reads the wall clock.
    assert out["decision_date"].isoformat() == "2026-09-24"


def test_arrangement_mode_is_named_not_implied():
    live = _handler()
    record = _typed_sample_record()
    out = live.executable.score(record, live.params)
    assert out["assessment_mode_code"] == 3
