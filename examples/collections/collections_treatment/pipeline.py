"""Collections treatment for one day's delinquent accounts: `build` is what `decider serve` runs.

Stages, in dependency order: account state, suspensions, score and bands, the matrix and its overlays, the
suspension check on the overlaid recommendation, the escalation path, the arrangement test and capacity.
`build` receives the matrix document and the six overlay stacks, one document each in `configs/<version>/`.
"""
from __future__ import annotations

from decider import DecisionTableConfig, dag

from collections_treatment import arrangements, capacity, matrix, path, scoring, state, suspensions
from collections_treatment.overlays import OverlayStack, in_dag

EMIT = (
    "account_id", "decision_date",
    # state
    "arrears_bucket_code", "balance_band_code", "balance_band_cell_id", "contact_band_code", "contact_band_cell_id",
    # suspensions
    "suspension_codes", "suspension_scopes", "suspension_sources", "suspension_expiries", "permitted_treatment_codes",
    "pre_prescription_flag",
    # score: adjusted and unadjusted
    "collections_score", "collections_score_unadjusted", "score_shift_adjustments_applied", "adjusted_scale",
    "scaling_change_adjustments_applied", "calibration_cell_id", "roll_probability_before_overlay", "roll_probability", "roll_probability_unadjusted",
    "odds_multiplier_adjustments_applied", "band_edge_shift", "band_boundary_shift_adjustments_applied",
    "collections_band_code", "collections_band_code_unadjusted", "collections_band_cell_id",
    "recovery_estimate", "cost_to_collect", "recovery_curve_cell_id",
    # matrix: as read, as overlaid, as finally applied
    "matrix_cell_id", "matrix_treatment_code", "matrix_treatment_intensity", "matrix_permitted_retries",
    "matrix_cooling_off_days", "overlaid_treatment_code", "treatment_code", "suspension_blocked", "treatment_intensity",
    "intensity_dial_adjustments_applied", "suppression_adjustments_applied",
    # path, as known on the day and as known now
    *path.PATH_OUTPUTS, *(f"{name}_as_at_now" for name in path.PATH_OUTPUTS),
    "path_floored_intensity", "must_escalate", "interval_or_cap_blocked", "interval_or_cap_detail",
    # arrangement
    "assessment_mode_code", "max_affordable_instalment", "discretionary_income", "net_monthly_income",
    "affordability_verdict_code", "arrangement_sustainable", "arrangement_residual", "arrangement_failure_detail",
    # capacity
    "pool_code", "selection_blocker_code", "allocated", "allocation_rank", "allocation_cutoff_rank",
    "allocation_ranking_basis", "non_selection_reason_code",
)


def build(
    treatment_matrix: DecisionTableConfig, score_shift: OverlayStack, scaling_change: OverlayStack,
    odds_multiplier: OverlayStack, band_boundary_shift: OverlayStack, intensity_dial: OverlayStack,
    treatment_suppression: OverlayStack,
):
    return dag(
        state.arrears_bucket_code, state.arrears_to_balance_ratio, state.build_balance_band_table(),
        scoring.build_contact_band_table(),
        suspensions.evaluate_suspensions_step, suspensions.pre_prescription_flag,
        scoring.build_scorecard(), in_dag(score_shift, "score_shift_adjustments_applied"),
        scoring.build_calibration_table(),
        in_dag(scaling_change, "scaling_change_adjustments_applied"), scoring.roll_probability_unadjusted,
        scoring.roll_probability_before_overlay, in_dag(odds_multiplier, "odds_multiplier_adjustments_applied"),
        scoring.clamp_probability_step, scoring.band_edge_origin_step,
        in_dag(band_boundary_shift, "band_boundary_shift_adjustments_applied"), scoring.collections_bands_step,
        scoring.collections_band_cell_id, scoring.build_recovery_curve_table(), scoring.recovery_and_cost_step,
        matrix.lookup(treatment_matrix), matrix.matrix_cell_id,
        in_dag(intensity_dial, "intensity_dial_adjustments_applied"), matrix.treatment_intensity_step,
        in_dag(treatment_suppression, "suppression_adjustments_applied"), matrix.overlaid_treatment_code_step,
        suspensions.treatment_after_suspensions_step,
        path.resolve_path_position_step, path.resolve_path_position_as_at_now, path.path_floored_intensity,
        path.interval_and_cap_check_step,
        arrangements.assessment_unit(), arrangements.arrangement_sustainability_step,
        capacity.build_pool_table(), capacity.selection_blocker_code, capacity.allocation_priority,
        capacity.allocate_capacity,
        name="collections_treatment",
    ).emit(*EMIT)
