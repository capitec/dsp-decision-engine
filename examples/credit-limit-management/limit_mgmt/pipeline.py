"""The per-account decision for the monthly programme: exclusions, behaviour score, matrix, caps, affordability, decrease.

`build(matrix)` is what `decider serve` runs for one account and what `simulation.run_cycle` runs over the whole book.
The budget that decides which proposed increases are actually funded is `allocation.allocate`, not a step: it needs
every account at once.
"""
from __future__ import annotations

from decider import dag, flow

from limit_mgmt import affordability, caps, decrease, exclusions, matrix as mx, outcome, population, scoring
from limit_mgmt.overlays import (
    apply_cycle_cap_step, matrix_dial_step, matrix_multiplier_excess_step, matrix_multiplier_step,
)

OUTPUTS = (
    "cycle_bucket_count", "revolving_utilisation_6m", "observed_spend_p90", "utilisation_band", "mob_band",
    "exclusion_codes", "is_excluded",
    "behaviour_score_raw", "behaviour_score", "behaviour_score_unadjusted", "score_adjustment_set_id",
    "score_adjustments_applied", "probability_of_default_raw", "probability_of_default",
    "probability_of_default_unadjusted", "pd_adjustment_set_id", "pd_adjustments_applied", "behaviour_grade",
    "behaviour_grade_cell_id", "behaviour_calibration_cell_id",
    "matrix_version", "matrix_cell_id", "matrix_multiplier_unadjusted", "matrix_max_increase_unadjusted",
    "matrix_min_increment_unadjusted", "matrix_multiplier_excess_unadjusted", "matrix_multiplier_excess_adjusted",
    "matrix_multiplier", "matrix_max_increase", "cycle_cap_bound", "matrix_adjustment_set_id",
    "matrix_adjustments_applied", "uncapped_target_limit", "uncapped_target_limit_unadjusted",
    "cap_product_max", "cap_income_multiple", "cap_total_exposure", "cap_observed_spend", "policy_proposed_limit",
    "policy_proposed_limit_unadjusted", "policy_binding_cap_code",
    "evidence_tier_code", "income_staleness_days", "notional_instalment", "buffer_adjustment_set_id",
    "buffer_adjustments_applied", "affordability_buffer_applied", "affordability_buffer_unadjusted",
    "gross_monthly_income", "net_monthly_income", "living_expenses", "existing_obligations", "discretionary_income",
    "max_affordable_instalment", "max_affordable_instalment_unadjusted", "affordability_verdict_code",
    "evidence_sufficiency_code", "adjustment_set_id", "adjustments_applied", "affordability_decline_reason_codes",
    "affordability_primary_reason_code", "affordability_reason_registry_version", "increase_path_code",
    "cap_affordability", "binding_cap_code",
    "decrease_trigger_fires", "decrease_target_limit", "decrease_notice_class_code", "decrease_trigger_codes",
    "final_proposed_limit", "change_type_code", "notice_class_code", "decline_reason_codes", "primary_reason_code",
    "reason_registry_version",
)


def build(matrix):
    """`matrix` is `configs/<version>/matrix.json`, loaded by `decider`; `matrix.build_matrix_table()` makes one in code."""
    matrix_unit = flow(
        mx.matrix_version_step, mx.lookup(matrix), matrix_multiplier_excess_step, matrix_dial_step,
        matrix_multiplier_step, apply_cycle_cap_step, mx.uncapped_target_limit_step,
        mx.uncapped_target_limit_unadjusted_step, name="matrix")
    return dag(
        population.unit, exclusions.unit, scoring.unit(), matrix_unit, caps.unit, affordability.unit(), decrease.unit,
        outcome.unit, name="limit_mgmt",
    ).emit(*OUTPUTS)
