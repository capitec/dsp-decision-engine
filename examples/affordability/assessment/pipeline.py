"""The affordability assessment as one pipeline, and as the two halves a repeated search needs.

`evidence_unit()` is everything that does not depend on the applicant's debts: household, income, tax and living
expenses. `capacity_unit()` is everything that does: obligations, discretionary income, capacity and the verdict.
`build()` is both, wired together; a search that varies only the debts scores `evidence_unit()` once and
`capacity_unit()` per candidate, from the same step objects.
"""
from __future__ import annotations

from decider import dag, flow

from credit_core import affordability, expense_norms

from assessment import capacity, debts, household, modes, verdict

EVIDENCE_OUTPUTS = (
    "gross_monthly_income", "income_source_code", "income_verification_tier", "income_haircut_applied",
    "income_variability_ratio", "statutory_deductions", "net_monthly_income", "applicant1_tax_table_version",
    "applicant1_tax_table_cell_id", "applicant2_tax_table_cell_id", "dependants_count", "living_expenses",
    "expense_basis_code", "norm_table_version", "statutory_norm_cell_id", "internal_norm_cell_id",
    "applicant_income_evidence_gap", "bureau_is_stale", "minimum_income_tier", "affordability_buffer_applied",
    "buffer_grid_cell_id", "residual_floor_amount", "residual_floor_cell_id",
)
CAPACITY_OUTPUTS = (
    "existing_obligations", "obligations_internal", "obligations_external", "total_exposure", "revolving_utilisation",
    "worst_arrears_months", "accounts_in_arrears_count", "obligation_account_type_codes", "obligation_treatment_codes",
    "obligation_monthly_amounts", "obligation_is_internal", "has_refer_account", "discretionary_income",
    "affordability_binding_constraint_code", "max_affordable_instalment_unadjusted", "max_affordable_instalment",
    "adjustment_set_id", "adjustments_applied", "evidence_sufficiency_code", "affordability_verdict_code",
    "discretionary_income_after", "decline_reason_codes", "primary_reason_code", "reason_registry_version",
)


def evidence_unit():
    """Household framing, income, tax, living expenses, buffer and floor lookups, and the evidence checks."""
    expense_norms_unit = flow(
        expense_norms.statutory_version_step, expense_norms.internal_version_step,
        expense_norms.build_statutory_table(), expense_norms.build_internal_table(),
        expense_norms.internal_norm_amount, expense_norms.norm_floor, expense_norms.norm_table_version,
        expense_norms.expense_basis_code, expense_norms.living_expenses, name="expense_norms")
    return dag(
        household.dependants_count, household.declared_living_expenses, household.statement_living_expenses,
        household.applicant_income("applicant1"), household.applicant_income("applicant2"),
        household.household_income,
        household.applicant1_tax_class, household.applicant2_tax_class,
        household.applicant_tax("applicant1"), household.applicant_tax("applicant2"),
        household.statutory_deductions, household.net_monthly_income,
        expense_norms_unit,
        household.applicant_income_evidence_gap, verdict.bureau_is_stale, modes.minimum_income_tier,
        capacity.build_buffer_grid(), capacity.build_residual_floor_table(),
        name="evidence",
    ).emit(*EVIDENCE_OUTPUTS)


def capacity_unit(capacity_adjustments=None, reason_codes=None):
    """Obligations, discretionary income, capacity (buffer against floor, then overlays) and the verdict.

    Reads the outputs of `evidence_unit()` by name, so pass them in beside the request. `capacity_adjustments` and
    `reason_codes` are config documents; the default is each one in this example's `configs/`.
    """
    return dag(
        debts.household_obligations,
        affordability.discretionary_income,
        capacity.buffer_capacity, capacity.floor_capacity, capacity.binding_capacity, capacity.overlay_unit(capacity_adjustments),
        verdict.evidence_sufficiency_code, verdict.proposal_verdict, verdict.affordability_verdict_code,
        verdict.discretionary_income_after, verdict.reasons_unit(reason_codes),
        name="capacity",
    ).emit(*CAPACITY_OUTPUTS)


def build(capacity_adjustments=None, reason_codes=None):
    """The whole assessment: what `decider serve` runs and what other projects embed.

    `capacity_adjustments` (overlays on the maximum instalment) and `reason_codes` are config documents; called
    without them, `build()` uses the ones in this example's `configs/`.
    """
    return dag(evidence_unit(), capacity_unit(capacity_adjustments, reason_codes), name="affordability_assessment").emit(
        *EVIDENCE_OUTPUTS, *CAPACITY_OUTPUTS, "court_ordered_deductions")
