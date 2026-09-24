"""Credit limit management (spec 07): the **per-account** pipeline --
§5.1 population/account state, §5.2 hard exclusions (a working-depth
subset), §5.3 behavioural scoring, §5.4 the limit assignment matrix (full
1 152 cells, loaded from `configs/<version>/matrix.json`), §5.5 caps,
§5.6 degraded-evidence affordability (through project 02, not forked),
§5.7 the decrease path (one trigger, D03), and the per-account half of
§5.9 (rounding, the minimum meaningful increase, reason codes).

This is what `decider build`/`decider serve` serve, and it is also what
runs in **batch** over the whole book (`batch_score()`, using the same
`Engine`/`.run()` call `decider` itself uses internally) to produce the
population-level DataFrame `limit_mgmt.allocation.run_allocation()`
consumes for §5.8. One implementation, three consumers: real-time (one
`.score()` call), the monthly batch (`.run()` over the book), and
simulation (`.run()` over a candidate snapshot/artefact set) -- see
`simulation.py` and NOTES.md "How I handled the portfolio-level budget".

§5.8 (the population-level budget allocation) is **not** part of this
`decider`-served pipeline -- see `limit_mgmt/allocation.py`'s docstring
for why a population-level decision cannot be expressed as a per-record
`decider` step, and `simulation.py` for how the two compose into one
monthly cycle.
"""
from __future__ import annotations

from datetime import date

import polars as pl
from decider import Engine, dag, flow, missing_as, param, step

from limit_mgmt import affordability, caps, decrease, exclusions, matrix as mx, outcome, population, reasons, scoring
from limit_mgmt.overlays import (
    apply_cycle_cap_step, behaviour_score_adjustment_step, matrix_dial_step, matrix_multiplier_excess_step,
    matrix_multiplier_step, pd_adjustment_step,
)


def _population_unit():
    return flow(
        population.utilisation_means, population.utilisation_band_step, population.mob_band_step,
        population.observed_spend, population.cycle_bucket_count_step,
        name="population",
    )


def _exclusions_unit():
    return flow(exclusions.exclusion_codes_step, exclusions.is_excluded_step, name="exclusions")


def _scoring_unit():
    pd_raw_step = scoring.probability_of_default_step.relabel(writes={"probability_of_default": "probability_of_default_raw"})
    return flow(
        scoring.build_behaviour_scorecard(),
        behaviour_score_adjustment_step,
        scoring.build_pd_calibration_table(),
        pd_raw_step,
        pd_adjustment_step,
        scoring.build_behaviour_grade_table(),
        name="scoring",
    )


def _matrix_unit(matrix_table):
    return flow(
        mx.matrix_version_step,
        matrix_table,
        matrix_multiplier_excess_step,
        matrix_dial_step,
        matrix_multiplier_step,
        apply_cycle_cap_step,
        mx.uncapped_target_limit_step,
        mx.uncapped_target_limit_unadjusted_step,
        name="matrix",
    )


def _caps_unit():
    return flow(
        caps.cap_product_max_step, caps.cap_income_multiple_step,
        caps.cap_total_exposure_step, caps.cap_observed_spend_step,
        caps.policy_proposed_limit_step, caps.policy_proposed_limit_unadjusted_step,
        name="caps",
    )


def _decrease_unit():
    return flow(
        decrease.decrease_trigger_fires_step, decrease.decrease_target_limit_step,
        decrease.decrease_notice_class_step, decrease.decrease_trigger_codes_step,
        name="decrease",
    )


def _outcome_unit():
    return flow(
        outcome.final_proposed_limit_step, outcome.change_type_code_step, outcome.notice_class_code_step,
        outcome.decline_reason_codes_step, reasons.REASON_REGISTRY.resolve_step(),
        name="outcome",
    )


def build(matrix):
    """`matrix` is loaded by `decider` from `configs/<version>/matrix.json`
    (`decider template`'s own `build(tree: ConfigurableStep)` shape, per BRIEF)."""
    matrix_table = matrix.relabel(writes={
        "multiplier": "matrix_multiplier_unadjusted", "max_increase": "matrix_max_increase_unadjusted",
        "min_increment": "matrix_min_increment_unadjusted", "cell_id": "matrix_cell_id",
    })

    return dag(
        _population_unit(),
        _exclusions_unit(),
        _scoring_unit(),
        _matrix_unit(matrix_table),
        _caps_unit(),
        affordability.affordability_assessment_unit(),
        _decrease_unit(),
        _outcome_unit(),
        name="limit_mgmt",
    ).emit(
        # `account_id`, `client_id` and `decision_id` are deliberately *not* named here:
        # they are top-level inputs no step reads, and naming an entirely-unread column
        # in an outer `.emit()` over already-`.emit()`-ed sub-units raises a spurious
        # `WiringError` (02 NOTES.md "Framework friction" 4.4, reproduced here). They
        # still appear in the output untouched, exactly as `flow`/`dag`'s own docstring
        # promises ("the output holds the input columns plus the values nothing reads").
        "decision_date", "product_code", "current_limit",
        "cycle_bucket_count", "revolving_utilisation_6m", "revolving_utilisation_3m", "observed_spend_p90",
        "utilisation_band", "mob_band",
        "exclusion_codes", "is_excluded",
        "behaviour_score_raw", "behaviour_score", "score_adjustment_set_id", "score_adjustments_applied",
        "probability_of_default_raw", "probability_of_default", "probability_of_default_unadjusted",
        "pd_adjustment_set_id", "pd_adjustments_applied", "behaviour_grade", "behaviour_grade_cell_id",
        "matrix_version", "matrix_cell_id",
        "matrix_multiplier_unadjusted", "matrix_max_increase_unadjusted", "matrix_min_increment_unadjusted",
        "matrix_multiplier_excess_unadjusted", "matrix_multiplier_excess_adjusted", "matrix_multiplier",
        "matrix_max_increase", "cycle_cap_bound", "matrix_adjustment_set_id", "matrix_adjustments_applied",
        "uncapped_target_limit", "uncapped_target_limit_unadjusted",
        "cap_product_max", "cap_income_multiple", "cap_total_exposure", "cap_observed_spend",
        "policy_proposed_limit", "policy_proposed_limit_unadjusted", "binding_cap_code",
        "notional_instalment", "evidence_tier_code", "income_staleness_days",
        "buffer_adjustment_set_id", "buffer_adjustments_applied", "affordability_buffer_applied",
        "affordability_buffer_unadjusted",
        "gross_monthly_income", "net_monthly_income", "living_expenses", "existing_obligations",
        "discretionary_income", "max_affordable_instalment", "max_affordable_instalment_unadjusted",
        "affordability_verdict_code", "evidence_sufficiency_code", "adjustment_set_id", "adjustments_applied",
        "affordability_decline_reason_codes", "affordability_primary_reason_code",
        "affordability_reason_registry_version",
        "cap_affordability", "increase_path_code",
        "decrease_trigger_fires", "decrease_target_limit", "decrease_notice_class_code", "decrease_trigger_codes",
        "final_proposed_limit", "change_type_code", "notice_class_code",
        "decline_reason_codes", "primary_reason_code", "reason_registry_version",
    )


def batch_score(engine: Engine, matrix_config, df: pl.DataFrame, params: dict | None = None) -> pl.DataFrame:
    """Runs the *same* per-account pipeline `build()` produces over a whole
    population DataFrame -- the "one implementation, three paths" answer
    (§10 item 3, README §2 hard part 2): production's monthly batch,
    simulation's candidate run, and the real-time path's single-record
    `decider serve` call all execute the identical bound pipeline, never a
    second expression of the matrix, caps or affordability logic. See
    `simulation.py`.
    """
    exe = engine.bind(build(matrix_config))
    return exe.run(df, params=params or {})
