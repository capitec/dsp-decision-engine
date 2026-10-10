"""Household framing: one or two applicants assessed as one household.

`credit_core`'s income and tax run once per applicant, relabelled onto that applicant's fields, and the results
are combined here. Applicant 2's fields are read only on a joint application.
"""
from __future__ import annotations

import math

from decikit import allow_fallback, flow, missing_as, param, step

from credit_core import deductions, income

# Per category the higher declaration counts for shared costs; every other category adds up.
SHARED_EXPENSES = frozenset({"accommodation", "water_electricity", "food", "insurance"})

_INCOME_READS = ("declared_income", "payslip_income", "statement_income", "statement_confidence",
                 "employer_confirmed_income", "bureau_estimated_income", "variable_pay_history")
_INCOME_WRITES = ("income_source_code", "income_verification_tier", "income_haircut_applied",
                  "gross_monthly_income", "income_variability_ratio")
_TAX_WRITES = ("statutory_deductions", "net_monthly_income", "tax_table_version", "tax_base", "tax_rate",
               "tax_band_lo", "tax_table_cell_id")


def applicant_income(prefix: str):
    """`credit_core.income.income_unit` on one applicant's fields: `applicant1_payslip_income` in, `applicant1_gross_monthly_income` out."""
    return income.income_unit.named(f"{prefix}_income").relabel(
        reads={name: f"{prefix}_{name}" for name in _INCOME_READS},
        writes={name: f"{prefix}_{name}" for name in _INCOME_WRITES})


def applicant1_tax_class(applicant1_employment_type_code: float) -> float:
    return applicant1_employment_type_code


def applicant2_tax_class(applicant2_employment_type_code: float = missing_as(0.0)) -> float:
    """Defaulted, because a solo application has no applicant 2."""
    return applicant2_employment_type_code


def applicant_tax(prefix: str):
    """Tax and statutory deductions on one applicant's own income. Tax is individual, so a household is never taxed on its sum."""
    unit = flow(deductions.tax_table_version_step, deductions.build_tax_table(), deductions.statutory_deductions,
                deductions.net_monthly_income, name=f"{prefix}_tax")
    return unit.relabel(
        reads={"gross_monthly_income": f"{prefix}_gross_monthly_income",
               "employment_type_code": f"{prefix}_tax_class"},
        writes={name: f"{prefix}_{name}" for name in _TAX_WRITES})


def dependants_count(
    is_joint_application: bool = missing_as(False),
    applicant1_dependants_count: float = missing_as(0.0), applicant2_dependants_count: float = missing_as(0.0),
) -> float:
    """Declared once per household: the higher of the two declarations."""
    return max(applicant1_dependants_count, applicant2_dependants_count if is_joint_application else 0.0)


def _consolidate(first: dict, second: dict) -> float:
    total = 0.0
    for category in first.keys() | second.keys():
        a, b = first.get(category) or 0.0, second.get(category) or 0.0
        total += max(a, b) if category in SHARED_EXPENSES else a + b
    return round(total, 2)


def _household_expenses(joint: bool, first: dict, second: dict) -> float:
    second = second if joint else {}
    return _consolidate(first, second) if first or second else math.nan  # NaN: nothing was supplied, which is not a zero


@allow_fallback
def declared_living_expenses(
    is_joint_application: bool = missing_as(False),
    applicant1_declared_expenses: dict = missing_as({}), applicant2_declared_expenses: dict = missing_as({}),
) -> float:
    return _household_expenses(is_joint_application, applicant1_declared_expenses, applicant2_declared_expenses)


@allow_fallback
def statement_living_expenses(
    is_joint_application: bool = missing_as(False),
    applicant1_statement_expenses: dict = missing_as({}), applicant2_statement_expenses: dict = missing_as({}),
) -> float:
    return _household_expenses(is_joint_application, applicant1_statement_expenses, applicant2_statement_expenses)


@allow_fallback
@step(outputs=("gross_monthly_income", "income_source_code", "income_verification_tier", "income_haircut_applied",
               "income_variability_ratio"))
def household_income(
    is_joint_application: bool,
    applicant1_gross_monthly_income: float, applicant1_income_source_code: int, applicant1_income_verification_tier: int,
    applicant1_income_haircut_applied: float, applicant1_income_variability_ratio: float,
    applicant2_gross_monthly_income: float, applicant2_income_source_code: int, applicant2_income_verification_tier: int,
    applicant2_income_haircut_applied: float, applicant2_income_variability_ratio: float,
    minor_contribution_threshold: float = param(0.10, ge=0.0, le=1.0),
) -> tuple[float, int, int, float, float]:
    """Household figures from each applicant's own (gross, source, tier, haircut, variability).

    Incomes add up after their own haircuts. The source code is the largest applicant's; the tier is the weakest
    among those contributing more than `minor_contribution_threshold`, so a small side income does not drag it down.
    """
    people = [(applicant1_gross_monthly_income, applicant1_income_source_code, applicant1_income_verification_tier,
               applicant1_income_haircut_applied, applicant1_income_variability_ratio)]
    if is_joint_application:
        people.append((applicant2_gross_monthly_income, applicant2_income_source_code,
                       applicant2_income_verification_tier, applicant2_income_haircut_applied,
                       applicant2_income_variability_ratio))
    gross = sum(p[0] for p in people)
    major = [p[2] for p in people if gross > 0 and p[0] / gross > minor_contribution_threshold] or [p[2] for p in people]
    before = sum(g / (1.0 - h) if h < 1.0 else g for g, _, _, h, _ in people)
    return (round(gross, 2), max(people, key=lambda p: p[0])[1], max(major),
            round(1.0 - gross / before, 4) if before > 0 else 0.0, max(p[4] for p in people))


def statutory_deductions(
    is_joint_application: bool, applicant1_statutory_deductions: float, applicant2_statutory_deductions: float,
) -> float:
    return round(applicant1_statutory_deductions + (applicant2_statutory_deductions if is_joint_application else 0.0), 2)


def net_monthly_income(
    is_joint_application: bool, applicant1_net_monthly_income: float, applicant2_net_monthly_income: float,
) -> float:
    return round(applicant1_net_monthly_income + (applicant2_net_monthly_income if is_joint_application else 0.0), 2)


def applicant_income_evidence_gap(
    is_joint_application: bool, applicant1_income_source_code: int, applicant2_income_source_code: int,
) -> int:
    """Whose income could not be established at all: 0 nobody, 1 or 2 that applicant, 3 both.

    A household is never assessed on one applicant's income alone, so any gap makes the verdict indeterminate.
    """
    return ((applicant1_income_source_code == income.NONE_ESTABLISHED)
            + 2 * (bool(is_joint_application) and applicant2_income_source_code == income.NONE_ESTABLISHED))
