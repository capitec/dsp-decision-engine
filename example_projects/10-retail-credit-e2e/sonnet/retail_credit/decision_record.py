"""The single decision record shape (spec 10 §5.31.1) and its assembly.

**One schema, not eight** (§5.31.1): every entry point's record uses this
same shape, sparse by construction -- a phase that did not run is *absent*,
carrying `record_completeness_code` 2, never collapsed into a null that
looks like "the value is zero" or "we could not tell".

Why this is a plain Python function over the pipeline's flat output, not a
decider step returning a nested struct: `pipeline.py`'s `build()` emits a
flat set of named columns (the pattern project 00 also uses, and the shape
`decider template` produces) rather than one nested-struct output, because
a `frame_step` (or, less certainly, a plain `step`) whose terminal output is
`list[struct]`-shaped crashes result materialisation in this framework
version (see project 00 NOTES.md "Framework friction" §4.2, and
`retail_credit/consolidation.py`'s docstring for where this project hits
the same thing). `build_decision_record` runs *after* `Engine.score`/`run`
has produced the flat row, entirely outside decider, and nests it into the
§5.31.2 shape for presentation, evidence storage or a client-facing
rendering. It is data assembly, not decision logic -- nothing in this
module narrows a value or chooses an outcome.
"""
from __future__ import annotations

from typing import Any

from retail_credit import entry_points, phases as phases_mod
from retail_credit.vocab import RecordCompletenessCode


def build_decision_record(row: dict[str, Any], entry_point_code: int, resolved_phase_ids: list[int]) -> dict[str, Any]:
    """Assemble the §5.31.2 floor from one scored row (a `dict`, as `Engine.score` returns)
    plus the phase set that actually ran. `row` already carries every field the pipeline
    emitted (see `pipeline.py`'s `.emit(...)` list) -- this only re-shapes it, it does not
    compute anything.
    """
    resolved_codes = {phases_mod.PHASES[pid].code for pid in resolved_phase_ids}
    absent_phases = [
        phases_mod.PHASES[pid].code
        for pid in phases_mod.PHASES
        if pid not in resolved_phase_ids and entry_point_code in phases_mod.PHASES[pid].entry_points
    ]

    return {
        # 1. Stable identifier, assigned before any logic ran (09 §5.15 item 1).
        "decision_id": row.get("decision_id"),
        # 2. entry_point_code, phase_set_id, decision_date, build identity.
        "entry_point_code": entry_point_code,
        "phase_set_id": entry_points.phase_set_id(entry_point_code, resolved_phase_ids),
        "phases_run": sorted(resolved_codes),
        "phases_absent_by_entry_point": absent_phases,
        "decision_date": row.get("decision_date"),
        # 3. Inputs as received -- the caller's own record; this project does not
        #    re-capture it here (that is a serving-layer concern, out of this slice).
        # 4-6. Table/cell/version attribution and the overlay stack.
        "table_versions": {
            "rate_card_cell_id": row.get("rate_card_cell_id"),
            "rate_card_version": row.get("rate_card_version"),
            "scorecard_id": row.get("scorecard_id"),
            "scorecard_version": row.get("scorecard_version"),
            "calibration_cell_id": row.get("calibration_cell_id"),
            "risk_grade_cell_id": row.get("risk_grade_cell_id"),
            "buffer_cell_id": row.get("buffer_cell_id"),
        },
        "adjustment_set_id": row.get("adjustment_set_id"),
        "adjustments_applied": row.get("adjustments_applied"),
        "score_unadjusted": row.get("score_unadjusted"),
        "probability_of_default_unadjusted": row.get("probability_of_default_unadjusted"),
        # 7-8. Cap chain, with every entry's verdict and status (P09's own evidence shape).
        "amount_cap": row.get("amount_cap"),
        "amount_cap_chain": row.get("amount_cap_chain"),
        "term_cap": row.get("term_cap"),
        "worst_acceptable_grade": row.get("worst_acceptable_grade"),
        # 9. Per-characteristic scorecard contributions.
        "score_contributions": row.get("score_contributions"),
        # 10. Every pricing evaluation the solve performed.
        "solve_evaluations": row.get("solve_evaluations"),
        "solve_binding_constraint": row.get("solve_binding_constraint"),
        # 11. Every loop pass (L1), its trigger, basis, result, discarded offers.
        "loop_pass_index": row.get("loop_pass_index", 0),
        "loop_termination_code": row.get("loop_termination_code"),
        "loop_passes": row.get("loop_passes"),
        # 12. value_basis_code / scenario_ref on values with more than one live version.
        "existing_obligations": row.get("existing_obligations"),
        "existing_obligations_basis_code": row.get("existing_obligations_basis_code"),
        "existing_obligations_hypothetical": row.get("existing_obligations_hypothetical"),
        "existing_obligations_hypothetical_scenario_ref": row.get("existing_obligations_hypothetical_scenario_ref"),
        # 13. Final-validation assertions.
        "validation_assertions": row.get("validation_assertions"),
        # 14. degraded_mode_code and per-source degradation state.
        "degraded_mode_code": row.get("degraded_mode_code"),
        "source_degradation_codes": row.get("source_degradation_codes"),
        # 15. The complete reason set, ranked, with registry version and primary.
        "decline_reason_codes": row.get("decline_reason_codes"),
        "primary_reason_code": row.get("primary_reason_code"),
        "reason_registry_version": row.get("reason_registry_version"),
        # 16. Per-phase timings -- not measured in this project's test harness; declared absent.
        "phase_budget_overrun_codes": row.get("phase_budget_overrun_codes", []),
        # Outcome.
        "outcome_code": row.get("outcome_code"),
        "record_completeness_code": row.get("record_completeness_code", int(RecordCompletenessCode.COMPLETE)),
    }
