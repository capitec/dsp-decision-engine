"""Flex Loan (product 10) granting and pricing -- spec 03.

Eleven stages (spec 03 §5), composed as one `dag`: eligibility gates ->
consent + fraud handoff -> bureau data-quality verdict -> application
scorecard/calibration/grading with four overlays -> the 52-rule cap
waterfall -> affordability (consumed from project 02) -> the bounded solve,
offer-set construction and final validation (one `frame_step`, see
`loan_granting/granting.py`) -> outcome resolution.

Every stage runs for every application (no `branch`-based short-circuit
after eligibility/fraud/DQ -- see `loan_granting/outcome.py`'s docstring);
`decider`'s `dag` resolves the real dependency order from what each step
reads and writes, regardless of the order they're listed here.
"""
from __future__ import annotations

from decider import dag, flow, param, step

from loan_granting import bureau_dq, eligibility, fraud, outcome, scoring, waterfall
from loan_granting.affordability import build_affordability_step
from loan_granting.granting import build_granting_step
from loan_granting.pricing import RateCardIndex
from loan_granting.reasons import REGISTRY

from credit_core.consent import consent_verdict_step


def statutory_ceiling(repo_rate: float = param(0.07, ge=0.0, le=0.30)) -> float:
    """The NCA margin-over-repo ceiling (spec 03 §5.7: 28.00% nominal at a 7.00% repo
    rate) -- a `param()` because Compliance moves it on gazette, mid-month, without a
    redeploy (03 §6.2)."""
    return round(repo_rate + 0.21, 4)


def recommendation_objective(recommendation_objective: str = param("largest_amount")) -> str:
    """Product's switch between the three offer-set objectives (03 §5.9), changed
    without a code release."""
    return recommendation_objective


def evaluation_ceiling_per_term(evaluation_ceiling_per_term: int = param(24, ge=1, le=64)) -> int:
    return evaluation_ceiling_per_term


statutory_ceiling_step = step(statutory_ceiling, output="statutory_ceiling")
recommendation_objective_step = step(recommendation_objective, output="recommendation_objective")
evaluation_ceiling_per_term_step = step(evaluation_ceiling_per_term, output="evaluation_ceiling_per_term")


def build(rate_card_flex_loan):
    rate_card_index = RateCardIndex.from_configurable_step(rate_card_flex_loan)

    return dag(
        eligibility.eligibility_gates,
        consent_verdict_step,
        fraud.fraud_handling_path_step, fraud.fraud_declined_step, fraud.fraud_forced_refer_step,
        fraud.fraud_bypass_applied_step,

        bureau_dq.bureau_availability_code_step, bureau_dq.data_quality_verdict_step,
        bureau_dq.bureau_is_stale_step, bureau_dq.is_thin_file_step, bureau_dq.bureau_referral_required_step,

        waterfall.cap_waterfall,

        scoring.build_scoring_unit(),

        build_affordability_step(),

        statutory_ceiling_step, recommendation_objective_step, evaluation_ceiling_per_term_step,
        build_granting_step(rate_card_index),

        flow(outcome.combined_decline_reason_codes_step, REGISTRY.resolve_step(), name="reasons"),
        outcome.application_outcome_code_step,
        outcome.referral_queue_code_step,

        name="loan_granting",
    ).emit(
        "eligibility_gate_ids", "eligibility_gate_verdicts", "is_eligible", "eligibility_decline_reasons",
        "consent_verdict", "fraud_handling_path", "fraud_bypass_applied",
        "bureau_availability_code", "data_quality_verdict", "bureau_is_stale", "is_thin_file",
        "bureau_referral_required",
        "scorecard_id", "segment_code", "score", "score_unadjusted", "score_reason_codes",
        "probability_of_default", "probability_of_default_unadjusted", "risk_grade", "risk_grade_cell_id",
        "risk_grade_version", "adjustment_set_id", "adjustments_applied",
        "amount_cap", "term_cap", "worst_acceptable_grade",
        "amount_cap_binding_rule_id", "term_cap_binding_rule_id", "worst_acceptable_grade_binding_rule_id",
        "amount_cap_chain_rule_ids", "amount_cap_chain_values",
        "waterfall_rule_ids", "waterfall_rule_owners", "waterfall_rule_ceilings", "waterfall_rule_status",
        "waterfall_decline_reason_codes", "uplift_authority_reference", "uplift_restrained_by_rule_id",
        "max_affordable_instalment", "discretionary_income", "affordability_verdict_code", "living_expenses",
        "existing_obligations", "affordability_norm_table_version", "affordability_adjustment_set_id",
        "offer_term_months", "offer_amounts", "offer_nominal_rates", "offer_instalments",
        "offer_total_costs_of_credit", "offer_effective_rates", "offer_is_recommended",
        "offer_binding_constraint_codes",
        "recommended_term", "recommended_amount", "recommended_instalment", "recommended_total_cost_of_credit",
        "recommended_effective_annual_rate", "recommended_binding_constraint_code",
        "suppressed_terms", "suppression_reason_codes", "solve_total_evaluations",
        "final_validation_passed", "final_validation_failures", "has_any_offer",
        "outcome_code", "referral_queue_code", "decline_reason_codes", "primary_reason_code",
        "reason_registry_version", "recommendation_objective", "statutory_ceiling",
    )
