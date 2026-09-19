"""P06(b) income, deductions and expenses — 34 decision points. Spec 5.7(b).

The first of the affordability chain's four links pulled FORWARD out of P10
(O-07: this must precede P07, because 14 of SC-A3's 61 characteristics are
income-derived). `exclusive_producer=True` on `net_monthly_income`
(values/register.py) makes a second producer anywhere in the flow — including
inside P10 — a build error, even where it would happen to agree.
"""

from __future__ import annotations

from decider2 import module, param, missing_as

def evidence_tier(income_evidence: dict, assessment_mode_code: int) -> int:
    """Seven tiers, 1 (employer-confirmed) to 7 (declared only). Tier
    selection differs by assessment_mode_code (origination vs limit vs
    campaign vs scenario) but the arithmetic below does not (P10's
    modes_share_arithmetic applies here too, one phase earlier)."""
    pass  # select the best available tier for this assessment mode

def haircut(evidence_tier: int, income_variability_ratio: float = missing_as(0.0),
           tenure_months: float = missing_as(999.0), segment_code: int = missing_as(0)) -> float:
    """Base haircut by tier, +4pp if variability > 0.28, +3pp if tenure < 9
    months, -2pp for segment 11."""
    pass  # base_haircut(evidence_tier) plus the three declared modifiers

def gross_monthly_income(income_evidence: dict, evidence_tier: int) -> float:
    pass  # the tier's own derivation of the gross figure

def net_monthly_income(gross_monthly_income: float, haircut: float,
                       decision_date: str) -> float:
    """THE single producer. Resolves statutory deductions from the 9 x 4
    effective-dated bracket table at decision_date. Eleven consumers
    (values/register.py) must read THIS value, never re-derive it."""
    pass  # apply haircut, then the statutory deduction table resolved at decision_date

def living_expenses(declared_expenses: float, net_monthly_income: float,
                    dependants_count: int, statement_derived_expenses: float = missing_as(0.0)
                    ) -> tuple[float, int]:
    """The HIGHER of declared, statement-derived, and the norm floor (15 income
    bands x 7 dependants x 2 components, stricter internal variant per
    product = 1 260 cells). Returns the value and which basis bound."""
    pass  # max(declared, statement_derived, norm_floor), record expense_basis_code

Determine = module(evidence_tier, haircut, gross_monthly_income, net_monthly_income,
                   living_expenses, name="income")
