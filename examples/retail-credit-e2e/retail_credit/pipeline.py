"""The retail credit decision for a new Flex Loan application, from admission to outcome.

`build` is what `decider build` and `decider serve` run. The dag resolves its steps in this order: admission, identity,
eligibility, bureau state and fraud; income, tax, expenses, obligations and segment; score, probability of default
and grade; policy caps; affordability; product routing; the consolidation scenario; pricing and the solve; the
offer; final validation; reasons and outcome.
"""
from __future__ import annotations

from decider import dag, step

from credit_core import deductions, expense_norms, income, instalment, obligations
from retail_credit import (
    affordability, caps, consolidation, disclosure, eligibility, features, flex_card, fraud, grading, intake, offer,
    pricing, routing, scoring, solve, validation,
)
from retail_credit.overlays import overlay_step


def rate_addon_base() -> float:
    return 0.0


# An add-on is the register applied to zero: the overlay's own amount, resolved once and then read by the solve and
# by the offer's price alike.
rate_addon_step = overlay_step(
    "nominal_annual_rate", "rate_addon", base_field="rate_addon_base", adjusted_output="rate_addon_bps",
    unadjusted_output="rate_addon_unadjusted")

EMIT = (
    # admission, identity, eligibility, fraud
    "p01_valid", "p01_rejection_reasons", "identity_resolution_path", "identity_confidence",
    "identity_is_trustworthy", "hard_eligibility_reasons", "is_eligible", "degraded_mode_code",
    "source_degradation_codes", "bureau_down_envelope_ok", "fraud_verdict_code", "fraud_reason_codes",
    "fraud_weighted_score", "fraud_rules_fired_count",
    # income, expenses, obligations, segment
    "gross_monthly_income", "income_source_code", "income_evidence_tier", "net_monthly_income", "statutory_deductions",
    "tax_table_cell_id", "living_expenses", "expense_basis_code", "norm_table_version", "existing_obligations",
    "obligations_internal", "obligations_external", "worst_arrears_months", "segment_code",
    # risk
    "score", "probability_of_default_unadjusted", "probability_of_default", "pd_adjustment_set_id",
    "pd_adjustments_applied", "calibration_cell_id", "risk_grade", "risk_grade_cell_id",
    # caps and routing
    "amount_cap_unadjusted", "amount_cap", "amount_cap_binding_rule", "amount_cap_chain", "amount_cap_adjustments_applied",
    "p09_declined", "p09_decline_reason", "term_cap", "worst_acceptable_grade", "product_10_eligible",
    "product_routing_reasons",
    # affordability, actual then hypothetical
    "discretionary_income", "residual_floor", "ratio_ceiling", "capacity", "pre_buffer", "affordability_buffer_unadjusted",
    "affordability_buffer_applied", "buffer_adjustments_applied", "max_affordable_instalment", "instalment_cap",
    "affordability_binding_constraint_code", "affordability_verdict_code", "consolidation_proposed", "existing_obligations_hypothetical",
    "max_affordable_instalment_hypothetical", "instalment_cap_hypothetical", "affordability_verdict_code_hypothetical",
    "offer_basis_code", "offer_affordability_verdict_code", "instalment_ceiling",
    # solve and price
    "rate_addon_bps", "rate_addon_adjustments_applied", "solved_amount", "solve_binding_constraint", "evaluated", "disqualified",
    "decision_open", "amount_grid", "offer_amount", "offer_binding_constraint", "has_offer", "offer_term_months", "rate_cell_id",
    "nominal_annual_rate_unadjusted", "nominal_annual_rate", "credit_life_rate", "initiation_fee", "initiation_fee_incl_tax",
    "amount_financed", "instalment_before_fees", "credit_life_premium", "credit_life_cell_id", "monthly_service_fee", "instalment",
    "total_cost_of_credit", "total_interest", "effective_annual_rate",
    # validation and outcome
    "validation_passed", "validation_failed_assertions", "decline_reason_codes", "primary_reason_code",
    "reason_registry_version", "outcome_code",
)


def build():
    return dag(
        intake.validate_request_step, intake.resolve_identity_step,
        eligibility.eligibility_gates_step, eligibility.hard_eligibility_reasons_step, eligibility.is_eligible_step,
        intake.source_degradation_step, fraud.evaluate_fraud_step,

        income.income_unit, features.income_evidence_tier_step,
        deductions.tax_table_version_step, deductions.build_tax_table(),
        deductions.statutory_deductions, deductions.net_monthly_income,
        expense_norms.statutory_version_step, expense_norms.internal_version_step,
        expense_norms.build_statutory_table(), expense_norms.build_internal_table(),
        expense_norms.internal_norm_amount, expense_norms.norm_floor, expense_norms.norm_table_version,
        expense_norms.expense_basis_code, expense_norms.living_expenses,
        obligations.obligations, features.segment_code_step, intake.bureau_down_within_envelope_step,

        scoring.build_scorecard(),
        grading.build_calibration_table(), grading.probability_of_default_raw_step,
        grading.probability_of_default_overlay_step, grading.build_risk_grade_table(),

        routing.product_range_step, caps.policy_caps_step, caps.amount_cap_unit,
        affordability.limits_unit(), affordability.capacity_unit(), routing.product_eligibility_step,
        consolidation.consolidation_scenario_step, consolidation.hypothetical_capacity_unit(),
        consolidation.offer_basis_step,

        step(pricing.offer_term_months), pricing.amount_independent_unit(),
        step(rate_addon_base), rate_addon_step, solve.solve_unit(),
        step(offer.decision_open), offer.assemble_offer_step, offer.priced_offer_step, step(flex_card.rate_cell_id),
        step(offer.total_cost_of_credit), step(offer.total_interest), instalment.effective_annual_rate_step,

        validation.revalidate_offer_step, disclosure.reasons_unit(), disclosure.outcome_code_step,
        name="retail_credit",
    ).emit(*EMIT)
