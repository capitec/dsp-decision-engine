"""A Flex Loan application: gates, fraud, bureau, score, caps, affordability, the offers and the outcome.

`decikit build` and `decikit serve` use `build`, which receives the rate card, the four overlay stacks and the
reason codes as the documents in `configs/<version>/`.
"""
from __future__ import annotations

from decikit import dag, flow, param, step

from assessment import modes
from assessment.pipeline import build as build_assessment
from credit_core import appetite, fees
from credit_core.vocab import SEGMENT_NEW_TO_BANK, SEGMENT_RETAIL_MASS

from loan_granting import bureau, eligibility, fraud, offers, outcome, scoring, waterfall


def statutory_ceiling(repo_rate: float = param(0.07, ge=0.0, le=0.30), margin_over_repo: float = param(0.21, ge=0.0)) -> float:
    """The highest nominal rate the law allows: a margin over the repo rate. Compliance moves it on gazette."""
    return round(repo_rate + margin_over_repo, 4)


def appetite_segment_code(segment_code: float) -> float:
    # workaround: credit_core's appetite grid has no new-to-bank rows, so they use the mass segment's
    return SEGMENT_RETAIL_MASS if segment_code == SEGMENT_NEW_TO_BANK else segment_code


def assessment_mode_code(mode: int = param(modes.NEW_APPLICATION, ge=1, le=4)) -> int:
    return mode


def build(rate_card_flex_loan, score_adjustments, calibration_scale_adjustments, probability_of_default_adjustments,
          probability_of_default_for_grading_adjustments, reason_codes):
    solve_terms, offer_set, final_validation = offers.steps(rate_card_flex_loan)

    # The assessment is one step. Its employment code is this flow's own, and the names it would write over this
    # flow's are moved aside.
    affordability = build_assessment().relabel(
        reads={"applicant1_employment_type_code": "employment_type_code"},
        writes={name: f"affordability_{name}" for name in (
            "decline_reason_codes", "primary_reason_code", "reason_registry_version", "adjustment_set_id",
            "adjustments_applied", "revolving_utilisation", "worst_arrears_months", "accounts_in_arrears_count")})

    return dag(
        eligibility.unit(),
        step(bureau.consent_verdict, output="consent_verdict"),
        bureau.bureau_availability_code, bureau.data_quality_verdict, bureau.is_thin_file, bureau.bureau_referral_required,
        fraud.fraud_handling_code_step, fraud.fraud_bypass_applied,

        scoring.unit(score_adjustments, calibration_scale_adjustments, probability_of_default_adjustments,
                     probability_of_default_for_grading_adjustments),
        appetite_segment_code,
        appetite.build_appetite_table().relabel(
            reads={"segment_code": "appetite_segment_code"},
            writes={"max_amount": "appetite_max_amount", "max_term": "appetite_max_term", "max_ratio": "appetite_max_ratio",
                    "min_price": "appetite_min_price", "cell_id": "appetite_cell_id"}),
        waterfall.employer_on_watchlist, waterfall.cap_waterfall,

        assessment_mode_code, affordability,

        statutory_ceiling, fees.fee_caps_version_step, fees.build_fee_caps_table(),
        offers.total_cost_ratio_threshold, offers.recommendation_objective,
        solve_terms, offer_set, final_validation,

        outcome.unit(reason_codes),
        name="loan_granting",
    ).emit(
        "eligibility_gate_verdicts", "is_eligible", "eligibility_decline_reasons", "consent_verdict",
        "fraud_handling_code", "fraud_bypass_applied", "bureau_availability_code", "data_quality_verdict",
        "bureau_is_stale", "is_thin_file", "bureau_referral_required",
        "scorecard_id", "segment_code", "score", "score_unadjusted", "score_reason_codes", "probability_of_default",
        "probability_of_default_unadjusted", "risk_grade", "risk_grade_cell_id", "risk_grade_version",
        "adjustment_set_id", "adjustments_applied", "appetite_cell_id", *waterfall.OUTPUTS,
        "max_affordable_instalment", "max_affordable_instalment_unadjusted", "discretionary_income",
        "affordability_verdict_code", "evidence_sufficiency_code", "living_expenses", "existing_obligations",
        "norm_table_version", "statutory_norm_cell_id", "internal_norm_cell_id", "buffer_grid_cell_id",
        "applicant1_tax_table_cell_id", "affordability_adjustment_set_id", "affordability_adjustments_applied",
        "statutory_ceiling", "fee_caps_cell_id", "initiation_fee_cap", "monthly_service_fee_cap",
        "recommendation_objective", "total_cost_ratio_threshold", "term_solutions", "solve_total_evaluations",
        "offers", "recommended_offer", "suppressed_terms", "suppression_reason_codes", "has_any_offer",
        "final_validation_passed", "final_validation_failures",
        "outcome_code", "referral_queue_code", "decline_reason_codes", "primary_reason_code", "reason_registry_version",
    )
