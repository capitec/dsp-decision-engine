"""A Flex Loan quote: eligibility, income, affordability, score, grade, price and outcome in one pipeline.

`decider build` and `decider serve` use `build`, which receives the rate card document
`configs/<version>/rate_card_flex_loan.json`.
"""
from __future__ import annotations

from datetime import date

from decider import allow_fallback, dag, flow, step

from credit_core import (
    affordability, calibration, deductions, eligibility, expense_norms, fees, income, instalment, obligations,
    rate_card, reason_codes, risk_grade, rounding, scorecard,
)
from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.vocab import OUTCOME_APPROVE, OUTCOME_DECLINE, OUTCOME_REFER

R_AFFORDABILITY_FAIL = 4001

REASON_REGISTRY = reason_codes.ReasonCodeRegistry("reasons-2026.09", [
    reason_codes.ReasonCode(eligibility.R_UNDERAGE, 10, "Applicant below minimum age", True),
    reason_codes.ReasonCode(eligibility.R_NOT_RESIDENT, 15, "Applicant not resident", True),
    reason_codes.ReasonCode(eligibility.R_NO_CAPACITY, 5, "No contractual capacity", True),
    reason_codes.ReasonCode(eligibility.R_PRODUCT_UNAVAILABLE, 40, "Product not available to applicant", False),
    reason_codes.ReasonCode(eligibility.R_EXCLUSION_LIST, 8, "On an internal exclusion list", True),
    reason_codes.ReasonCode(eligibility.R_SANCTIONED, 1, "Sanctioned party", True),
    reason_codes.ReasonCode(eligibility.R_DECEASED, 2, "Deceased on record", True),
    reason_codes.ReasonCode(eligibility.R_DEBT_REVIEW, 12, "Under debt review", True),
    reason_codes.ReasonCode(eligibility.R_ADMINISTRATION, 12, "Under administration", True),
    reason_codes.ReasonCode(R_AFFORDABILITY_FAIL, 20, "Proposed instalment exceeds affordable capacity", True),
])

# Two tighten-only score overlays. An applicant in segment 3 who came through channel 4 gets both, in stack order.
ADJUSTMENT_SET_ID = "AS-2026.09"
SCORE_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-2026-001", kind="score_shift", target="score", effect=AdjustmentEffect("add", -18.0),
        scope={"segment_code": 3}, stack_position=1, owner="Credit Risk Policy", approval_reference="CRC-2026-014",
        rationale="New-to-bank segment defaulting above model prediction for two quarters",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True),
    Adjustment(
        adjustment_id="ADJ-2026-002", kind="score_shift", target="score", effect=AdjustmentEffect("add", -5.0),
        scope={"channel_code": 4}, stack_position=2, owner="Credit Risk Policy", approval_reference="CRC-2026-019",
        rationale="Broker channel 4 showing early-life deterioration",
        effective_from=date(2026, 6, 1), effective_to=date(2027, 6, 1), review_date=date(2027, 1, 1), tighten_only=True),
])


@allow_fallback
def decline_reasons(eligibility_decline_reasons: list[int], affordability_verdict_code: int) -> list[int]:
    if affordability_verdict_code == affordability.FAIL:
        return [*eligibility_decline_reasons, R_AFFORDABILITY_FAIL]
    return eligibility_decline_reasons


def outcome_code(is_eligible: bool, affordability_verdict_code: int) -> int:
    """Decline when ineligible or unaffordable, refer when affordability is marginal or cannot be assessed."""
    if not is_eligible or affordability_verdict_code == affordability.FAIL:
        return OUTCOME_DECLINE
    if affordability_verdict_code in (affordability.MARGINAL, affordability.INDETERMINATE):
        return OUTCOME_REFER
    return OUTCOME_APPROVE


def build(rate_card_flex_loan):
    # The instalment is computed in a flow, because `instalment` is written twice: fees added, then rounded.
    instalment_unit = flow(
        instalment.instalment_before_fees_step, instalment.instalment_step,
        rounding.round_instalment_step.relabel(reads={"amount": "instalment"}),
        instalment.total_cost_of_credit_step, instalment.total_interest_step, instalment.effective_annual_rate_step,
        name="instalment",
    ).emit("instalment_before_fees", "instalment")

    # `decline_reason_codes` is written twice, as fired and then ranked, so these two also form a flow.
    reasons_unit = flow(step(decline_reasons, output="decline_reason_codes"), REASON_REGISTRY.resolve_step(),
                        name="reasons").emit("decline_reason_codes")

    return dag(
        step(eligibility.decline_reason_codes, output="eligibility_decline_reasons"),
        eligibility.is_eligible_step.relabel(reads={"decline_reason_codes": "eligibility_decline_reasons"}),

        income.income_unit,
        deductions.tax_table_version_step, deductions.build_tax_table(),
        deductions.statutory_deductions, deductions.net_monthly_income,

        expense_norms.statutory_version_step, expense_norms.internal_version_step,
        expense_norms.build_statutory_table(), expense_norms.build_internal_table(),
        expense_norms.internal_norm_amount, expense_norms.norm_floor, expense_norms.norm_table_version,
        expense_norms.expense_basis_code, expense_norms.living_expenses,

        obligations.obligations,

        affordability.discretionary_income, affordability.affordability_buffer_applied,
        affordability.max_affordable_instalment,
        affordability.affordability_verdict_code_step.relabel(reads={"proposed_instalment": "instalment"}),

        scorecard.build_scorecard().relabel(writes={"score": "score_raw"}),
        SCORE_ADJUSTMENTS.apply_stack_step("score", ADJUSTMENT_SET_ID, base_field="score_raw"),
        calibration.build_calibration_table(), calibration.probability_of_default_step,
        risk_grade.build_risk_grade_table(),

        rounding.round_advance_step.relabel(reads={"amount": "requested_amount"}),
        rate_card_flex_loan.relabel(writes={"cell_id": "rate_cell_id"}), rate_card.rate_card_rate_step,
        rate_card.out_of_range,

        fees.fee_caps_version_step, fees.build_fee_caps_table(),
        fees.initiation_fee_step, fees.initiation_fee_capped_step, fees.monthly_service_fee_step,
        instalment_unit,

        reasons_unit,
        outcome_code,
        name="flex_loan_quote",
    ).emit(
        "gross_monthly_income", "income_source_code", "income_haircut_applied", "net_monthly_income",
        "statutory_deductions", "tax_table_cell_id", "living_expenses", "expense_basis_code", "norm_table_version",
        "statutory_norm_cell_id", "internal_norm_cell_id", "existing_obligations", "obligations_internal",
        "obligations_external", "worst_arrears_months", "discretionary_income", "affordability_buffer_applied",
        "max_affordable_instalment", "affordability_verdict_code", "score_raw", "score", "score_unadjusted",
        "adjustment_set_id", "adjustments_applied", "calibration_cell_id", "calibration_version",
        "probability_of_default", "risk_grade", "risk_grade_boundary_lo", "risk_grade_boundary_hi",
        "risk_grade_cell_id", "risk_grade_version", "offered_amount", "nominal_annual_rate", "rate_cell_id",
        "rate_card_version", "out_of_range", "fee_caps_cell_id", "initiation_fee", "monthly_service_fee",
        "instalment", "total_cost_of_credit", "total_interest", "effective_annual_rate", "eligibility_decline_reasons",
        "is_eligible", "decline_reason_codes", "primary_reason_code", "reason_registry_version", "outcome_code",
    )
