"""The evidence waterfall (s5.6). Tiers A-E, haircuts, staleness.

This is the whole of the "degraded evidence" answer, and the point of putting
it here is that `core.affordability` is NOT touched. s10.14 requires that the
difference between origination and programme assessment be visible entirely in
inputs and evidence. So:

  * project 02 supplies `gross_monthly_income` from a payslip with a 0% haircut
    and `income_staleness_days = 3`;
  * this project supplies it from a deposit pattern with a 15% haircut and
    `income_staleness_days = 780`;
  * `core.affordability` is the same module object in both pipelines, with a
    different `buffer` param.

The seam that a careless implementation breaks is the one where this file grows
an `if is_programme:`. The guard against that is FRAMEWORK-DEMANDS #9: the
evidence outputs below are DECLARED and NON-DROPPABLE, so a consumer cannot
take the income and leave the provenance behind.
"""

from decider2 import module, panel, param, table
from decider2.credit import core

EvidenceTiers = table(
    "clm.evidence_tiers",
    keys=("tier_code",),
    values={"haircut": float, "max_staleness_days": int, "automatic_eligible": bool,
            "indexation_cap_months": int},
    domain={"tier_code": range(1, 6)},       # A=1 .. E=5
    dense=True, effective_dated=True,
    owner="Credit Risk Policy + Compliance",
)


def salary_pattern_confidence(
    deposit_months_observed: int,
    deposit_variability: float,
    variability_ceiling: float = param(0.15, ge=0, le=1.0,
                                       description="tier A month-to-month variability"),
) -> float:
    """How well the transactional-account credits match a salary pattern."""
    pass


def evidence_tier_code(
    salary_pattern_confidence: float,
    deposit_months_observed: int,
    has_transactional_account: bool,
    declared_income_age_months: int,
    tables,
) -> int:
    """The ordered waterfall. A verified salary deposits (74%); B irregular
    deposits (9%); C declared income refreshed within 12 months (6%); D declared
    income stale, median age 26 months (9%); E no usable evidence (2%).

    Tiers D and E are not usable for an automatic increase. They are not a
    failure -- they route to the conditional path (s5.6)."""
    pass


def income_staleness_days(evidence_tier_code: int, deposit_last_seen_day: int,
                          declared_income_captured_day: int, shared) -> int:
    """Age of the income basis AT decision_date, from the basis actually used."""
    pass


def gross_monthly_income_c(
    evidence_tier_code: int,
    deposit_mean_6m_c: int,
    declared_gross_income_c: int,
    income_staleness_days: int,
    tables,
    cpi_rate: float = param(0.052, ge=0, le=0.25,
                            description="indexation applied to tier C, capped at 12 months"),
) -> int:
    """The income figure, after the tier's haircut and, for tier C, after CPI
    indexation to a maximum of twelve months."""
    pass


def income_haircut_applied(evidence_tier_code: int, tables) -> float:
    """0% on tier A, 15% on tier B's trailing 6-month mean."""
    return tables.evidence_tiers.haircut[
        tables.evidence_tiers.cell(evidence_tier_code)]


IncomeEvidence = module(
    salary_pattern_confidence, evidence_tier_code, income_staleness_days,
    gross_monthly_income_c, income_haircut_applied,
    name="income_evidence",
    evidence=[
        "evidence_tier_code", "income_staleness_days", "income_haircut_applied",
        "deposit_account_id", "deposit_months_observed", "deposit_window_start_day",
        "deposit_window_end_day", "declared_income_captured_day",
    ],
)

# The expense side. `core.expense_norms` unchanged; the indexation and the
# 36-month discard are this project's inputs to it.
ExpenseBasis = module(
    core.expense_norms.indexed_declared,
    core.expense_norms.norm_floor,
    core.expense_norms.binding_basis,
    name="expense_basis",
    evidence=["expense_basis_code", "norm_table_version", "indexation_months"],
).bind(discard_declared_after_months=36)

# The obligations side is genuinely better evidence than the income side, and
# asymmetrically so: the Bank sees new external debt but not a new salary.
Obligations = core.obligations.at(
    inputs={"account_list": "bureau_account_list"},
).with_evidence(["bureau_file_id", "bureau_as_of_date", "obligations_delta_pct"])
