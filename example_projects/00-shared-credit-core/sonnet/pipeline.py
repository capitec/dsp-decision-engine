"""credit-core demo: a Flex Loan (product 10) affordability + pricing + risk quote.

Not a product flow of its own -- projects 02/03/10 own those. This wires a
representative slice of `credit_core`'s published capabilities into one
pipeline so `decider build`/`decider serve` prove they compose: eligibility
-> income -> deductions -> expense norms -> obligations -> affordability
-> scorecard (with one overlay pair demonstrating stack-on/off) ->
calibration -> risk grade -> rate card (the full 63 360-cell Flex Loan
grid, loaded from `configs/<version>/rate_card_flex_loan.json` so it can be
refreshed without a redeploy, 00 §9) -> fees -> instalment -> reason codes
-> outcome.

`core.bureau`, `core.eligibility`'s bureau-derived flags, `core.appetite`,
`core.exposure`, `core.consent`, `core.credit_life` and
`core.adverse_events` are exercised directly in `tests/` instead of wired
in here (every capability is standalone-testable by design, §7.5) --
wiring all twenty-two into one demo pipeline would multiply the
column-naming bookkeeping below for no extra proof of the mechanism.
"""
from __future__ import annotations

from datetime import date

from decider import dag, flow, step

from credit_core import (
    affordability, deductions, eligibility, expense_norms, fees, income, instalment, obligations, rate_card,
    reason_codes, risk_grade, rounding, scorecard as sc,
)
from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.calibration import build_calibration_table, probability_of_default_step

OUTCOME_APPROVE = 1
OUTCOME_APPROVE_WITH_CONDITIONS = 2
OUTCOME_REFER = 3
OUTCOME_DECLINE = 4

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

# Two overlays on `score`, both tighten-only, demonstrating composition order and scope
# collision (change scenario 13): a new-to-bank segment shift and a channel-level shift both
# apply when segment_code=3 and channel_code=4 meet in one application.
ADJUSTMENT_SET_ID = "AS-2026.09"
SCORE_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-2026-001", kind="score_shift", target="score",
        effect=AdjustmentEffect("add", -18.0), scope={"segment_code": 3}, stack_position=1,
        owner="Credit Risk Policy", approval_reference="CRC-2026-014",
        rationale="New-to-bank segment defaulting above model prediction for two quarters",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True,
    ),
    Adjustment(
        adjustment_id="ADJ-2026-002", kind="score_shift", target="score",
        effect=AdjustmentEffect("add", -5.0), scope={"channel_code": 4}, stack_position=2,
        owner="Credit Risk Policy", approval_reference="CRC-2026-019",
        rationale="Broker channel 4 showing early-life deterioration",
        effective_from=date(2026, 6, 1), effective_to=date(2027, 6, 1), review_date=date(2027, 1, 1),
        tighten_only=True,
    ),
])


def _reasons_unit():
    """Eligibility gate reasons, plus an affordability-fail reason, ranked through the registry."""

    def combine_decline_reasons(eligibility_decline_reasons: list[int], affordability_verdict_code: int) -> list[int]:
        reasons = list(eligibility_decline_reasons)
        if affordability_verdict_code == affordability.FAIL:
            reasons.append(R_AFFORDABILITY_FAIL)
        return reasons

    combine_step = step(combine_decline_reasons, output="decline_reason_codes")
    return flow(combine_step, REASON_REGISTRY.resolve_step(), name="reasons").emit("decline_reason_codes")


def _instalment_unit():
    """Amortisation, fees folded in, then rounded to the cent -- a same-name waterfall, so
    it has to be a `flow`, not a `dag` (two members would otherwise both write "instalment")."""
    return flow(
        instalment.instalment_before_fees_step,
        instalment.instalment_step,
        rounding.round_instalment_step.relabel(reads={"amount": "instalment"}),
        instalment.total_cost_of_credit_step,
        instalment.total_interest_step,
        instalment.effective_annual_rate_step,
        name="instalment",
    ).emit("instalment_before_fees")


def outcome_code(is_eligible: bool, affordability_verdict_code: int) -> int:
    if not is_eligible:
        return OUTCOME_DECLINE
    if affordability_verdict_code == affordability.FAIL:
        return OUTCOME_DECLINE
    if affordability_verdict_code in (affordability.MARGINAL, affordability.INDETERMINATE):
        return OUTCOME_REFER
    return OUTCOME_APPROVE


def build(rate_card_flex_loan):
    rate_card_table = rate_card_flex_loan.relabel(writes={"cell_id": "rate_card_cell_id"})

    eligibility_reasons_step = step(eligibility.decline_reason_codes, output="eligibility_decline_reasons")
    is_eligible_step = eligibility.is_eligible_step.relabel(reads={"decline_reason_codes": "eligibility_decline_reasons"})

    scorecard_step = sc.build_scorecard().relabel(writes={"score": "score_raw"})
    score_adjustment_step = SCORE_ADJUSTMENTS.apply_stack_step("score", ADJUSTMENT_SET_ID, base_field="score_raw")

    risk_grade_table = risk_grade.build_risk_grade_table()
    risk_grade_output_step = step(risk_grade.risk_grade_output, output="risk_grade")

    offered_amount_step = rounding.round_advance_step.relabel(reads={"amount": "requested_amount"})

    return dag(
        eligibility_reasons_step, is_eligible_step,

        income.gross_monthly_income, income.income_source_code, income.income_verification_tier,
        income.income_haircut_applied, income.income_variability_ratio,

        deductions.tax_table_version_step, deductions.build_tax_table(), deductions.statutory_deductions,
        deductions.net_monthly_income,

        expense_norms.statutory_version_step, expense_norms.internal_version_step,
        expense_norms.build_statutory_table(), expense_norms.build_internal_table(),
        expense_norms.internal_norm_amount, expense_norms.norm_floor, expense_norms.norm_table_version,
        expense_norms.expense_basis_code, expense_norms.living_expenses,

        obligations.obligations,

        affordability.discretionary_income, affordability.affordability_buffer_applied,
        affordability.max_affordable_instalment, affordability.affordability_verdict_code,

        scorecard_step, score_adjustment_step,
        build_calibration_table(), probability_of_default_step,
        risk_grade_table, risk_grade_output_step,

        offered_amount_step, rate_card_table, rate_card.rate_card_rate_step, rate_card.out_of_range,

        fees.initiation_fee_step, fees.initiation_fee_capped_step, fees.monthly_service_fee_step,

        _instalment_unit(),

        _reasons_unit(),
        outcome_code,

        name="credit_core_demo",
    ).emit(
        "gross_monthly_income", "income_source_code", "net_monthly_income", "statutory_deductions",
        "living_expenses", "expense_basis_code", "norm_table_version",
        "existing_obligations", "obligations_internal", "obligations_external", "worst_arrears_months",
        "discretionary_income", "affordability_buffer_applied", "max_affordable_instalment",
        "affordability_verdict_code",
        "score_raw", "score", "score_unadjusted", "adjustment_set_id", "adjustments_applied",
        "probability_of_default", "risk_grade", "risk_grade_boundary_lo", "risk_grade_boundary_hi",
        "offered_amount", "nominal_annual_rate", "rate_card_cell_id", "rate_card_version", "out_of_range",
        "initiation_fee", "monthly_service_fee",
        "instalment", "total_cost_of_credit", "total_interest", "effective_annual_rate",
        "eligibility_decline_reasons", "is_eligible", "decline_reason_codes", "primary_reason_code",
        "reason_registry_version", "outcome_code",
    )
