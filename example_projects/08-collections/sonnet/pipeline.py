"""Collections treatment assignment (spec 08). `build()` returns the one pipeline this
project serves, in the shape `decider template` produces (see SERVE.md).

Composed from `credit_core` (project 00, read-only via `PYTHONPATH`), project 02's
affordability assessment (also read-only, loaded through `collections_treatment.
arrangements` to avoid the `pipeline.py`-name collision documented there), decider's
built-ins (`DecisionTableConfig`, `ScorecardConfig`, `frame_step`), and
`collections_treatment`'s own modules for what spec 08 declares locally (§4.1):
account state, suspensions, the collections score and matrix, the escalation path,
arrangement sustainability and capacity allocation.
"""
from __future__ import annotations

from decider import dag, flow

from collections_treatment import arrangements, capacity_alloc, matrix, path, scoring, state, suspensions


def _risk_and_state():
    return dag(
        state.arrears_bucket_code_step,
        state.arrears_to_balance_ratio_step,
        state.build_balance_band_table(),
        name="account_state",
    )


def _suspensions_unit():
    return dag(
        suspensions.evaluate_suspensions_step,
        suspensions.permitted_treatment_codes_step,
        suspensions.suspended_blocks_all_step,
        suspensions.pre_prescription_flag_step,
        name="suspensions",
    )


def _scoring_unit():
    # `build_calibration_table()` and `scaling_change_step` both write `scale` -- the
    # "scaling change" overlay overwrites the table's own value by design (§5.3), which
    # `dag()` refuses to order on its own ("collections_calibration_segments and
    # scaling_change both write 'scale'; use flow(...)... the later one winning" --
    # `decider.exceptions.WiringError`). `flow()` is exactly 02's own precedent for this
    # same pattern (`assessment/capacity.py`'s `overlay_unit`).
    calibration_and_scale = flow(
        scoring.build_calibration_table(), scoring.scaling_change_step, name="calibration",
    )
    return dag(
        scoring.build_scorecard().relabel(reads={"current_bucket": "arrears_bucket_code"}),
        scoring.score_shift_step,
        calibration_and_scale,
        scoring.roll_probability_unadjusted_step,
        scoring.odds_multiplier_step,
        scoring.band_boundary_shift_step,
        scoring.build_band_table(),
        scoring.build_unadjusted_band_table(),
        scoring.build_contact_band_table(),
        scoring.build_recovery_curve_table(),
        scoring.recovery_and_cost_step,
        name="scoring",
    )


def _matrix_unit():
    return dag(
        matrix.build_treatment_matrix(),
        matrix.intensity_dial_step,
        matrix.treatment_suppression_step,
        name="matrix",
    )


def _path_unit():
    """The escalation path, run once for the as-known-on-the-day decision (drives today's
    actual treatment) and a second time, relabelled onto the `_as_at_now` variants of the
    same five event flags, for the as-at-now comparison (§5.5, §10 item 4). Same function,
    two settings -- the "one capability, run twice" pattern 00/02 both use."""
    as_known = path.resolve_path_position_step
    as_at_now = path.resolve_path_position_step.relabel(
        reads={
            "qualifying_payment": "qualifying_payment_as_at_now",
            "promise_captured": "promise_captured_as_at_now",
            "cured": "cured_as_at_now",
            "dispute_raised": "dispute_raised_as_at_now",
            "arrangement_activated": "arrangement_activated_as_at_now",
        },
        writes={
            "path_position": "path_position_as_at_now", "episode_open": "episode_open_as_at_now",
            "episode_opened_delta": "episode_opened_delta_as_at_now",
            "intensity_ceiling": "intensity_ceiling_as_at_now",
            "channel_attempt_count": "channel_attempt_count_as_at_now",
            "reset_event": "reset_event_as_at_now", "escalation_or_hold": "escalation_or_hold_as_at_now",
        },
    ).named("resolve_path_position_as_at_now")
    interval_and_cap_check_step = path.interval_and_cap_check_step.relabel(
        reads={"permitted_retries": "matrix_permitted_retries", "cooling_off_days": "matrix_cooling_off_days"},
    )
    return dag(
        as_known, as_at_now,
        path.apply_intensity_floor_step,
        interval_and_cap_check_step,
        name="path",
    )


def _arrangement_unit():
    """§5.6: every arrangement tested through project 02's `core.affordability`, in
    ARRANGEMENT mode, plus this project's own sustainability test on top (§5.6's
    "distressed mode... a parameterisation of one capability, not a fork")."""
    return dag(
        arrangements.assessment_mode_code_step,
        arrangements.arrangement_affordability_unit(),
        arrangements.arrangement_sustainability_step,
        name="arrangement",
    )


def build():
    core = dag(
        _risk_and_state(),
        _suspensions_unit(),
        _scoring_unit(),
        _matrix_unit(),
        _path_unit(),
        _arrangement_unit(),
        name="collections_core",
    )
    return dag(
        core,
        capacity_alloc.allocate_capacity(),
        name="collections_treatment",
    ).emit(
        # Identity and attribution (09 §5.15 minimum: decision id, stable logic ids).
        # `decision_id` is deliberately not named here: it is a top-level input nothing in
        # this graph reads, and naming an untouched input in an outer `.emit()` over nested
        # `dag()`s raises `WiringError: emit('decision_id'): no step produces 'decision_id'
        # and it is not a declared input column` even though it still appears in the output
        # untouched -- see 02 NOTES.md "Framework friction" 4.4, reproduced identically here.
        # `client_id` is the same story: nothing in this graph reads it either.
        "account_id", "decision_date",
        # Account state.
        "arrears_bucket_code", "balance_band_code", "balance_band_cell_id",
        # Suspensions (§5.2) -- the full, individually-attributed set, never the first match.
        "suspension_codes", "suspension_scopes", "suspension_blocks_all", "suspension_sources",
        "suspension_expiries", "permitted_treatment_codes", "suspended_blocks_all", "pre_prescription_flag",
        # Score (§5.3) -- adjusted and unadjusted, every overlay kind, per-characteristic
        # contributions are on the scorecard's own output columns (read back by the caller).
        "collections_score", "collections_score_unadjusted", "score_shift_adjustments_applied",
        "scale", "scaling_change_adjustments_applied", "calibration_cell_id",
        "roll_probability_before_overlay", "roll_probability", "odds_multiplier_adjustments_applied",
        "roll_probability_for_banding", "band_boundary_shift_adjustments_applied",
        "collections_band_code", "collections_band_cell_id",
        "collections_band_code_unadjusted", "collections_band_cell_id_unadjusted",
        "contact_band_code", "contact_band_cell_id",
        "recovery_estimate", "cost_to_collect", "recovery_curve_cell_id",
        # Matrix (§5.4) -- as read and as overlaid, plus attribution.
        "matrix_treatment_code", "matrix_treatment_intensity", "matrix_permitted_retries",
        "matrix_cooling_off_days", "matrix_cell_id",
        "treatment_code", "treatment_intensity", "intensity_dial_adjustments_applied",
        "suppression_adjustments_applied",
        # Path (§5.5) -- as-known (drives today) and as-at-now (comparison).
        "path_position", "episode_open", "intensity_ceiling", "channel_attempt_count",
        "reset_event", "escalation_or_hold", "path_floored_intensity",
        "path_position_as_at_now", "episode_open_as_at_now", "intensity_ceiling_as_at_now",
        "channel_attempt_count_as_at_now", "reset_event_as_at_now", "escalation_or_hold_as_at_now",
        "must_escalate", "interval_or_cap_blocked", "interval_or_cap_detail",
        # Arrangement (§5.6).
        "assessment_mode_code", "max_affordable_instalment", "discretionary_income", "net_monthly_income",
        "affordability_verdict_code", "arrangement_sustainable", "arrangement_residual",
        "arrangement_failure_detail",
        # Capacity allocation (§5.9).
        "pool", "allocated", "allocation_pool", "allocation_rank", "allocation_cutoff_rank",
        "allocation_ranking_basis", "non_selection_reason_code",
    )
