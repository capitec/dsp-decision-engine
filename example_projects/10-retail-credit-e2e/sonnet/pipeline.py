"""retail_credit demo pipeline: entry point 1 (new credit application), product 10 only.

The one real path SCOPE.md asks this project to flesh out end to end:
P01 -> P02 -> P03 -> P04 -> P05 -> P06 -> P07 -> P08 -> P09a -> P10 ->
P09b -> P11 -> P12 -> P13 -> [loop L1] -> P16 -> P17 -> P18, built entirely
from `credit_core` (project 00 §6) plus this project's own modules (10
§1.1: "the only thing it takes from outside is the shared library's
published capabilities"). See `retail_credit/ordering.py` for how this
order maps onto §5.22's declared constraints, and each phase module's own
docstring for what "real" means at this slice's depth versus what is
declared and left as a stub (`retail_credit.limit_assignment`,
`retail_credit.entry_points`' rows for entry points 2-8).

Column-naming discipline follows project 00's own convention (its
NOTES.md "What I publish"): every table-backed capability's `cell_id`
output is relabelled to `<capability>_cell_id` before it joins this dag,
because a second table's bare `cell_id` write silently collides with the
first's.
"""
from __future__ import annotations

from decider import dag, flow, param, step

from credit_core import deductions, eligibility, expense_norms, income, obligations as obligations_mod
from credit_core.consent import consent_verdict
from credit_core.credit_life import credit_life_cap_applied_step, credit_life_premium_step
from credit_core.fees import fee_caps_version_step, initiation_fee_capped_step, initiation_fee_step
from credit_core.instalment import (
    effective_annual_rate_step, instalment_before_fees_step, instalment_step, total_cost_of_credit_step,
    total_interest_step,
)
from credit_core.rounding import round_instalment_step

from retail_credit import (
    acquisition, affordability_phase, cap_waterfall, consolidation, disclosure, eligibility_consent, features,
    fraud, grading, identity, offers, p01_routing, pricing, routing, scoring, solve, validation,
)
from retail_credit.overlays import ADJUSTMENT_SET_ID, OVERLAY_REGISTER


def _p03_unit():
    """P03 -- consent and hard eligibility (10 §5.4): every gate evaluated, none
    short-circuited (the "short-circuit tension", 10 §5.4)."""
    eligibility_reasons_step = step(eligibility.decline_reason_codes, output="eligibility_decline_reasons")
    return dag(
        eligibility_reasons_step,
        eligibility_consent.product_10_extra_reasons_step,
        eligibility_consent.combine_eligibility_reasons_step,
        eligibility.is_eligible_step.relabel(reads={"decline_reason_codes": "hard_eligibility_reasons"}),
        eligibility_consent.consent_verdict_step,
        name="p03",
    )


def _p06_unit():
    """P06 -- feature derivation (10 §5.7): income/deductions/expense-norms/obligations are
    `core.*` units, wired exactly as project 00's own demo pipeline wires them (the
    "hidden coupling" DEPS.md documents, 10 §5.7(b)) -- plus this project's own segment
    assignment and evidence-tier calibration.
    """
    return dag(
        income.gross_monthly_income, income.income_source_code, income.income_verification_tier,
        income.income_haircut_applied, income.income_variability_ratio,
        features.income_evidence_tier_step,

        deductions.tax_table_version_step, deductions.build_tax_table(), deductions.statutory_deductions,
        deductions.net_monthly_income,

        expense_norms.statutory_version_step, expense_norms.internal_version_step,
        expense_norms.build_statutory_table(), expense_norms.build_internal_table(),
        expense_norms.internal_norm_amount, expense_norms.norm_floor, expense_norms.norm_table_version,
        expense_norms.expense_basis_code, expense_norms.living_expenses,

        obligations_mod.obligations,

        features.assign_segment_step,
        name="p06",
    )


def _p08_unit():
    """P08 -- calibration, grading and adjustments (10 §5.9), O-01: adjustments before
    grading. The overlay register is resolved **once**, here (O-05/O-21) -- every later
    phase that reads an overlaid value (P09's cap reduction, P10's buffer, P12's rate
    add-on) reads from `retail_credit.overlays.OVERLAY_REGISTER`, never re-resolves it.
    """
    return dag(
        scoring.build_scorecard_a3(),
        grading.build_calibration_table().relabel(writes={"cell_id": "calibration_cell_id"}),
        grading.probability_of_default_step,
        grading.probability_of_default_adjustment_step(),
        grading.build_risk_grade_table().relabel(writes={"cell_id": "risk_grade_cell_id"}),
        grading.risk_grade_output_step,
        name="p08",
    )


def _p09a_unit():
    """P09, first pass (O-10): amount/term/grade entries, before P10, then the cap overlay
    (10 §5.10's worked example: "last moved by an overlay")."""
    return dag(
        cap_waterfall.evaluate_amount_and_grade_caps_step,
        cap_waterfall.amount_cap_overlay_step().relabel(
            writes={"adjustment_set_id": "amount_cap_adjustment_set_id",
                    "adjustments_applied": "amount_cap_adjustments_applied"}),
        name="p09a",
    )


def _p10_unit():
    """P10 -- affordability (10 §5.11): 00's `discretionary_income` unit, this project's
    own capacity/ratio-ceiling/buffer composition, 00's `max_affordable_instalment`
    arithmetic reused by `.relabel()`, this project's own verdict policy.
    """
    from credit_core.affordability import discretionary_income_step
    return dag(
        discretionary_income_step,
        affordability_phase.residual_floor_step,
        affordability_phase.capacity_step,
        affordability_phase.build_ratio_ceiling_table().relabel(writes={"cell_id": "ratio_ceiling_cell_id"}),
        affordability_phase.ratio_ceiling_step,
        affordability_phase.pre_buffer_step,
        affordability_phase.buffer_channel_code_step,
        affordability_phase.build_buffer_table().relabel(
            reads={"channel_code": "buffer_channel_code"}, writes={"cell_id": "buffer_cell_id"}),
        affordability_phase.buffer_adjustment_step().relabel(
            writes={"adjustment_set_id": "buffer_adjustment_set_id", "adjustments_applied": "buffer_adjustments_applied"}),
        affordability_phase.max_affordable_instalment_step,
        affordability_phase.affordability_verdict_code_step,
        name="p10",
    )


def _p12_unit():
    """P12 -- pricing (10 §5.13), O-14: rate -> fee -> premium -> instalment. Prices
    `offer_amount` (P16's assembled offer), the one place this decision's rate, fee,
    premium and instalment are computed for real (O-22: rounding applied at exactly one
    place -- `offer_amount` is already R250-grid-aligned by the solve, so no second
    rounding step runs here).
    """
    rate_card_table = pricing.load_rate_card().relabel(writes={"cell_id": "rate_card_cell_id"})
    credit_life_table = pricing.load_credit_life_table().relabel(
        writes={"cell_id": "credit_life_cell_id", "rate": "credit_life_rate_per_1000"})

    priced_term_step = pricing.priced_term_months_step.relabel(reads={"term_months": "offer_term_months"})
    long_term_loading_step = pricing.long_term_loading_step.relabel(reads={"term_months": "offer_term_months"})

    initiation_fee_offer = initiation_fee_step.relabel(reads={"offered_amount": "offer_amount"})
    initiation_fee_capped_offer = initiation_fee_capped_step

    financed_annuity = instalment_before_fees_step.relabel(
        reads={"offered_amount": "amount_financed", "term_months": "offer_term_months"})
    financed_premium = credit_life_premium_step.relabel(
        reads={"offered_amount": "amount_financed", "rate": "credit_life_rate_per_1000"})

    instalment_unit = flow(
        financed_annuity,
        instalment_step,
        round_instalment_step.relabel(reads={"amount": "instalment"}),
        total_cost_of_credit_step.relabel(reads={"term_months": "offer_term_months"}),
        total_interest_step.relabel(reads={"offered_amount": "offer_amount"}),
        effective_annual_rate_step,
        name="instalment",
    ).emit("instalment_before_fees")

    return dag(
        step(offer_term_months, output="offer_term_months"),
        rate_card_table.relabel(reads={"offered_amount": "offer_amount"}),
        priced_term_step,
        pricing.rate_card_rate_step,
        long_term_loading_step,
        pricing.add_long_term_loading_step,
        pricing.rate_overlay_step().relabel(
            writes={"adjustment_set_id": "rate_adjustment_set_id", "adjustments_applied": "rate_adjustments_applied"}),
        pricing.statutory_ceiling_ok_step,

        initiation_fee_offer,
        initiation_fee_capped_offer,
        pricing.initiation_fee_incl_tax_step,
        pricing.amount_financed_step.relabel(reads={"offered_amount": "offer_amount"}),

        credit_life_table,
        financed_premium,
        credit_life_cap_applied_step.relabel(
            reads={"offered_amount": "amount_financed", "rate": "credit_life_rate_per_1000"}),

        step(monthly_service_fee, output="monthly_service_fee"),

        instalment_unit,
        name="p12",
    )


def offer_term_months(term_months: int, term_cap: float) -> int:
    return int(min(term_months, term_cap))


def monthly_service_fee(fee: float = param(87.98, ge=0.0)) -> float:
    """10 §5.13(b): R76.50 excluding tax, R87.98 including -- a flat, statutorily capped
    fee this project keeps as one `param()` (the inclusive figure) rather than reusing
    `core.fees.monthly_service_fee`'s own flat-fee arithmetic, since that function's
    parameter names (`flat_fee`/`cap`) both default to the same value and this project's
    number is the tax-inclusive one throughout (10 §5.13(d)'s instalment formula).
    """
    return fee


def _l1_seed_step():
    """Seeds `loop()`'s carries (`retail_credit.consolidation.l1_loop`) before entering the
    loop -- `loop_converged` starts `True` (skip the loop entirely) unless affordability
    actually failed *and* the client is consolidation-eligible (10 §5.20: P14 is
    conditional on entry point 1 exactly on that condition), matching the declared
    `resolve_phase_set` logic in `retail_credit.entry_points`.
    """
    def seed_l1_state(
        affordability_verdict_code: int, is_consolidation_eligible: bool, existing_obligations: float,
        discretionary_income: float, solved_amount: float, solved_instalment: float,
        solve_binding_constraint: str,
    ) -> tuple[int, bool, float, int, float, int, float, int, float, float, str]:
        from retail_credit.affordability_phase import FAIL
        from retail_credit.vocab import ValueBasisCode
        already_ok = affordability_verdict_code != FAIL or not is_consolidation_eligible
        return (
            0, already_ok, existing_obligations, int(ValueBasisCode.ACTUAL), discretionary_income, 0,
            0.0, 0, solved_amount, solved_instalment, solve_binding_constraint,
        )

    return step(seed_l1_state, outputs=(
        "loop_pass_index", "loop_converged", "existing_obligations_hypothetical",
        "existing_obligations_basis_code", "discretionary_income_hypothetical",
        "existing_obligations_hypothetical_scenario_ref", "max_affordable_instalment_hypothetical",
        "loop_termination_code", "loop_solved_amount", "loop_solved_instalment", "loop_binding_constraint",
    ))


def _p18_unit():
    reasons = flow(
        disclosure.combine_decline_reasons_step, disclosure.rank_reasons_step, name="reasons",
    ).emit("decline_reason_codes", "primary_reason_code", "reason_registry_version")
    return dag(reasons, disclosure.outcome_code_step, name="p18")


def build():
    return dag(
        p01_routing.validate_request_step,

        identity.resolve_identity_step,

        _p03_unit(),

        acquisition.orchestrate_acquisition_step,
        acquisition.bureau_down_reduced_envelope_ok_step,

        fraud.evaluate_fraud_step,

        _p06_unit(),

        _p08_unit(),

        _p09a_unit(),
        _p10_unit(),
        cap_waterfall.evaluate_instalment_cap_step,

        routing.product_10_eligible_step,

        pricing.zero_base_step,
        pricing.resolve_rate_addon_bps_step().relabel(
            writes={"adjustment_set_id": "rate_addon_adjustment_set_id",
                    "adjustments_applied": "rate_addon_adjustments_applied"}),

        solve.solve_step_wired,

        flow(_l1_seed_step(), consolidation.l1_loop(), name="l1_seeded").emit(
            "loop_pass_index", "loop_termination_code", "existing_obligations_hypothetical",
            "existing_obligations_basis_code", "existing_obligations_hypothetical_scenario_ref",
            "max_affordable_instalment_hypothetical", "loop_solved_amount", "loop_solved_instalment",
            "loop_binding_constraint",
        ),

        offers.assemble_offer_step,

        _p12_unit(),

        validation.revalidate_offer_step,

        _p18_unit(),

        name="retail_credit_ep1_product10",
    ).emit(
        "p01_valid", "p01_rejection_reasons",
        "identity_resolution_path", "identity_confidence", "identity_is_trustworthy",
        "hard_eligibility_reasons", "is_eligible", "consent_verdict_code",
        "degraded_mode_code", "source_degradation_codes", "bureau_down_envelope_ok",
        "fraud_verdict_code", "fraud_reason_codes", "fraud_weighted_score", "fraud_rules_fired_count",
        "gross_monthly_income", "income_source_code", "net_monthly_income", "statutory_deductions",
        "income_evidence_tier", "living_expenses", "expense_basis_code", "norm_table_version",
        "existing_obligations", "obligations_internal", "obligations_external", "worst_arrears_months",
        "segment_code",
        "score", "probability_of_default", "probability_of_default_unadjusted",
        "adjustment_set_id", "adjustments_applied", "risk_grade",
        "amount_cap", "amount_cap_binding_rule", "p09_declined", "p09_decline_reason", "term_cap",
        "worst_acceptable_grade", "instalment_cap",
        "product_10_eligible", "product_routing_reasons",
        "discretionary_income", "capacity", "ratio_ceiling", "pre_buffer",
        "affordability_buffer_applied", "affordability_buffer_unadjusted", "max_affordable_instalment",
        "affordability_verdict_code",
        "offer_term_months", "rate_card_cell_id", "rate_card_version", "nominal_annual_rate",
        "nominal_annual_rate_unadjusted", "statutory_ceiling_ok",
        "initiation_fee", "initiation_fee_capped", "initiation_fee_incl_tax", "amount_financed",
        "credit_life_premium", "monthly_service_fee",
        "instalment", "total_cost_of_credit", "total_interest", "effective_annual_rate",
        "solved_amount", "solved_instalment", "solve_binding_constraint", "solve_rate_card_cell_id",
        "loop_pass_index", "loop_termination_code", "existing_obligations_hypothetical",
        "existing_obligations_basis_code", "existing_obligations_hypothetical_scenario_ref",
        "max_affordable_instalment_hypothetical",
        "offer_amount", "offer_instalment", "offer_binding_constraint", "has_offer",
        "validation_passed", "validation_failed_assertions", "rederived_instalment",
        "decline_reason_codes", "primary_reason_code", "reason_registry_version",
        "outcome_code",
    )
