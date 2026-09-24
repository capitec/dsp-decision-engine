"""The affordability and obligations assessment (spec 02): single and joint
applicants, four assessment modes, three shapes of answer, one arithmetic
path underneath all of them.

Composed entirely from `credit_core` (project 00): the five affordability
units (`income`, `deductions`, `expense_norms`, `obligations`,
`affordability`) plus the library-wide mechanisms (`adjustments`,
`reason_codes`, `evidence`). Everything in `assessment/` is what DEPS.md's
"Cycles" §2 assigns to 02: household framing, the four modes, the verdict,
the three answer shapes -- composed from 00's units, never re-deriving them.

Two entry points, built from the *same* step objects (spec 02 §5.7.2 item 3,
"the separation [must not become] a second entry point that can drift from
the first"):

- `evidence_unit()` -- stages 1-4 (household framing, income, deductions,
  living expenses). Does not depend on the account list.
- `capacity_unit()` -- stages 5-7 (obligations, discretionary income,
  capacity, verdict). The only stage project 06's 400-scenario search needs
  to re-run per scenario; see `tests/test_scenario_budget.py`.

`build()` wires both into the one real-time pipeline for a new application.
"""
from __future__ import annotations

from datetime import date

from decider import dag, flow, step

from credit_core import deductions, expense_norms, reason_codes, rounding
from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.affordability import discretionary_income

from assessment import capacity, evidence_gates, household, modes, verdict

# --- Reason codes (09 §5.15 item 11: declared codes, drawn from a versioned registry) -----

R_APPLICANT1_INCOME_UNESTABLISHED = 201
R_APPLICANT2_INCOME_UNESTABLISHED = 202
R_BOTH_APPLICANTS_INCOME_UNESTABLISHED = 203
R_BUREAU_STALE = 204
R_REFER_ACCOUNT = 205
R_INCOME_BELOW_MINIMUM_TIER = 206
R_NO_NET_INCOME = 207
R_INSTALMENT_EXCEEDS_CAPACITY = 208

REASON_REGISTRY = reason_codes.ReasonCodeRegistry("affordability-reasons-2026.09", [
    reason_codes.ReasonCode(R_APPLICANT1_INCOME_UNESTABLISHED, 10, "Applicant 1 income could not be established", True),
    reason_codes.ReasonCode(R_APPLICANT2_INCOME_UNESTABLISHED, 10, "Applicant 2 income could not be established", True),
    reason_codes.ReasonCode(R_BOTH_APPLICANTS_INCOME_UNESTABLISHED, 5, "Neither applicant's income could be established", True),
    reason_codes.ReasonCode(R_BUREAU_STALE, 30, "Bureau view older than the permitted window", True),
    reason_codes.ReasonCode(R_REFER_ACCOUNT, 25, "An account type requires manual review", True),
    reason_codes.ReasonCode(R_INCOME_BELOW_MINIMUM_TIER, 20, "Income evidence below the product's minimum tier", True),
    reason_codes.ReasonCode(R_NO_NET_INCOME, 15, "No net income established", True),
    reason_codes.ReasonCode(R_INSTALMENT_EXCEEDS_CAPACITY, 40, "Proposed instalment exceeds the affordable maximum", False),
])

_EVIDENCE_REASON = {
    verdict.EVIDENCE_APPLICANT1_INCOME_UNESTABLISHED: R_APPLICANT1_INCOME_UNESTABLISHED,
    verdict.EVIDENCE_APPLICANT2_INCOME_UNESTABLISHED: R_APPLICANT2_INCOME_UNESTABLISHED,
    verdict.EVIDENCE_BOTH_APPLICANTS_INCOME_UNESTABLISHED: R_BOTH_APPLICANTS_INCOME_UNESTABLISHED,
    verdict.EVIDENCE_BUREAU_STALE: R_BUREAU_STALE,
    verdict.EVIDENCE_REFER_ACCOUNT: R_REFER_ACCOUNT,
    verdict.EVIDENCE_INCOME_BELOW_MINIMUM_TIER: R_INCOME_BELOW_MINIMUM_TIER,
    verdict.EVIDENCE_NO_NET_INCOME: R_NO_NET_INCOME,
}


def _reason_codes(evidence_sufficiency_code: int, affordability_verdict_code: int) -> list[int]:
    reasons = []
    reason = _EVIDENCE_REASON.get(evidence_sufficiency_code)
    if reason is not None:
        reasons.append(reason)
    if affordability_verdict_code == verdict.FAIL:
        reasons.append(R_INSTALMENT_EXCEEDS_CAPACITY)
    return reasons


def _reasons_unit():
    combine = step(_reason_codes, output="decline_reason_codes")
    return flow(combine, REASON_REGISTRY.resolve_step(), name="reasons")


# --- Overlays on `max_affordable_instalment` (§5.6.2: tighten-only, i.e. may only lower it) -

ADJUSTMENT_SET_ID = "AS-02-2026.09"
CAPACITY_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-02-2026-001", kind="cap_adjustment", target="max_affordable_instalment",
        effect=AdjustmentEffect("multiply", 0.90), scope={"product_code": 10}, stack_position=1,
        owner="Credit Risk Policy", approval_reference="CRC-2026-031",
        rationale="Product 10 early-life delinquency above appetite for two quarters",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True,
    ),
])


# --- Stage grouping -------------------------------------------------------------------------

def evidence_unit():
    """Stages 1-4 (household framing, income, deductions, living expenses). Everything
    project 06's 400-scenario loop computes once and holds fixed (02 §5.7.2 item 3)."""
    expense_norms_unit = flow(
        expense_norms.statutory_version_step, expense_norms.internal_version_step,
        expense_norms.build_statutory_table(), expense_norms.build_internal_table(),
        expense_norms.internal_norm_amount, expense_norms.norm_floor, expense_norms.norm_table_version,
        expense_norms.expense_basis_code, expense_norms.living_expenses,
        name="expense_norms",
    )
    return dag(
        household.dependants_count,
        household.consolidate_declared_expenses, household.consolidate_statement_expenses,

        household.applicant1_income_unit, household.applicant2_income_unit,
        household.household_gross_monthly_income_step, household.household_income_source_code_step,
        household.household_income_verification_tier_step, household.household_income_haircut_applied_step,
        household.household_income_variability_ratio_step,

        household.applicant1_deductions_unit, household.applicant2_deductions_unit,
        household.statutory_deductions, household.net_monthly_income,

        expense_norms_unit,

        household.applicant_income_evidence_gap,
        evidence_gates.bureau_is_stale,
        modes.minimum_income_tier,

        capacity.build_buffer_grid(), capacity.build_residual_floor_table(),

        name="evidence",
    ).emit(
        "gross_monthly_income", "income_source_code", "income_verification_tier", "income_haircut_applied",
        "income_variability_ratio", "statutory_deductions", "net_monthly_income",
        "living_expenses", "expense_basis_code", "norm_table_version",
        "dependants_count", "applicant_income_evidence_gap", "bureau_is_stale", "minimum_income_tier",
        "affordability_buffer_applied", "buffer_grid_cell_id", "residual_floor_amount", "residual_floor_cell_id",
        "statutory_norm_cell_id", "internal_norm_cell_id",
        "declared_living_expenses", "statement_living_expenses",
    )


def capacity_unit():
    """Stages 5-7 (obligations, discretionary income, capacity, verdict). The stage
    project 06 re-runs per scenario -- reads the `evidence_unit()` outputs by name, never
    redoing them (02 §5.7.2 item 3)."""
    overlay_unit = flow(
        CAPACITY_ADJUSTMENTS.apply_stack_step(
            "max_affordable_instalment", ADJUSTMENT_SET_ID, base_field="max_affordable_instalment_before_overlay",
        ),
        rounding.round_instalment_step.relabel(
            reads={"amount": "max_affordable_instalment"}, writes={"instalment": "max_affordable_instalment"},
        ),
        name="overlay",
    )
    return dag(
        household.household_obligations,

        discretionary_income,
        capacity.max_affordable_instalment_before_overlay_step,
        overlay_unit,

        verdict.evidence_sufficiency_code,
        verdict.affordability_verdict_code,
        verdict.discretionary_income_after,

        _reasons_unit(),

        name="capacity",
    ).emit(
        "existing_obligations", "obligations_internal", "obligations_external", "total_exposure",
        "revolving_utilisation", "worst_arrears_months", "accounts_in_arrears_count",
        "obligation_account_type_codes", "obligation_treatment_codes", "obligation_monthly_amounts",
        "obligation_is_internal", "has_refer_account",
        "discretionary_income", "affordability_binding_constraint_code",
        "max_affordable_instalment_before_overlay", "max_affordable_instalment_unadjusted",
        "max_affordable_instalment", "adjustment_set_id", "adjustments_applied",
        "evidence_sufficiency_code", "affordability_verdict_code", "discretionary_income_after",
        "decline_reason_codes", "primary_reason_code", "reason_registry_version",
    )


def build():
    return dag(evidence_unit(), capacity_unit(), name="affordability_assessment").emit(
        "gross_monthly_income", "income_source_code", "income_verification_tier", "income_haircut_applied",
        "income_variability_ratio",
        "statutory_deductions", "court_ordered_deductions", "net_monthly_income",
        "living_expenses", "expense_basis_code", "norm_table_version",
        "existing_obligations", "obligations_internal", "obligations_external",
        "total_exposure", "revolving_utilisation", "worst_arrears_months", "accounts_in_arrears_count",
        "discretionary_income", "max_affordable_instalment", "max_affordable_instalment_unadjusted",
        "affordability_buffer_applied", "affordability_binding_constraint_code",
        "affordability_verdict_code", "evidence_sufficiency_code",
        "adjustment_set_id", "adjustments_applied",
        "decline_reason_codes", "primary_reason_code", "reason_registry_version",
        "discretionary_income_after",
        "dependants_count",
    )
