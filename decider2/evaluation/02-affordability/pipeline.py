"""
Affordability Assessment Pipeline

Implements affordability assessment for credit decisioning.
"""

from decider2 import flow, param


def gross_monthly_income(
    gross_income_raw: float,
    employment_type_code: int,
    evidence_tier: int = 2,
    income_haircut_pct: float = param(5.0, ge=0.0, le=55.0),
) -> float:
    """Apply haircut to gross income based on evidence tier."""
    haircut = income_haircut_pct
    if evidence_tier == 6:
        haircut += 15.0
    elif evidence_tier == 5:
        haircut += 10.0
    haircut = min(haircut, 55.0)
    return gross_income_raw * (100.0 - haircut) / 100.0


def income_tax_monthly(
    gross_monthly_income: float,
    applicant_age_years: int,
) -> float:
    """Calculate monthly income tax."""
    annual = gross_monthly_income * 12.0
    if annual <= 95_750:
        tax = 0.0
    elif annual <= 237_100:
        tax = (annual - 95_750) * 0.18
    else:
        tax = 25_410 + (annual - 237_100) * 0.26
    rebate = 9_444.0
    if applicant_age_years >= 65:
        rebate += 2_736.0
    tax = max(0.0, tax - rebate)
    return tax / 12.0


def unemployment_insurance_monthly(
    gross_monthly_income: float,
    is_pensioner_or_grant: bool = False,
    ui_ceiling: float = param(177.12, ge=0.0),
) -> float:
    """Calculate UIF contribution."""
    if is_pensioner_or_grant:
        return 0.0
    uif = (gross_monthly_income * 1.0) / 100.0
    return min(uif, ui_ceiling)


def statutory_deductions(
    income_tax_monthly: float,
    unemployment_insurance_monthly: float,
    court_ordered_deductions: float,
) -> float:
    """Total statutory deductions."""
    return income_tax_monthly + unemployment_insurance_monthly + court_ordered_deductions


def net_monthly_income(
    gross_monthly_income: float,
    statutory_deductions: float,
) -> float:
    """Net income after deductions."""
    return gross_monthly_income - statutory_deductions


def living_expenses_declared(
    expense_declaration_rand: float,
) -> float:
    """Declared living expenses."""
    return max(0.0, expense_declaration_rand)


def living_expenses_norm(
    gross_monthly_income: float,
    dependants_count: int,
) -> float:
    """Statutory minimum expense norm."""
    fixed = 1_000.0
    if gross_monthly_income < 10_000:
        fixed = 800.0
    elif gross_monthly_income < 50_000:
        fixed = 3_675.38
    else:
        fixed = 6_592.88
    
    dep_mult = 1.0 + (float(dependants_count) * 0.18)
    fixed = fixed * dep_mult
    norm = fixed + max(0.0, gross_monthly_income - 10_000.0) * 6.75 / 100.0
    return norm


def living_expenses(
    living_expenses_declared: float,
    living_expenses_norm: float,
) -> float:
    """Select higher of declared and norm."""
    return max(living_expenses_declared, living_expenses_norm)


def existing_obligations_external(
    bureau_account_count: int,
    avg_monthly_instalment_bureau: float,
) -> float:
    """Bureau account obligations."""
    return float(max(0, bureau_account_count)) * avg_monthly_instalment_bureau


def existing_obligations_internal(
    internal_account_count: int,
    avg_monthly_instalment_internal: float,
) -> float:
    """Internal account obligations."""
    return float(max(0, internal_account_count)) * avg_monthly_instalment_internal


def existing_obligations(
    existing_obligations_external: float,
    existing_obligations_internal: float,
) -> float:
    """Total obligations."""
    return existing_obligations_external + existing_obligations_internal


def discretionary_income(
    net_monthly_income: float,
    living_expenses: float,
    court_ordered_deductions: float,
    existing_obligations: float,
) -> float:
    """Discretionary income after all deductions."""
    return net_monthly_income - living_expenses - court_ordered_deductions - existing_obligations


def max_affordable_instalment_unadjusted(
    discretionary_income: float,
    dependants_count: int,
    buffer_percentage: float = param(10.0, ge=5.0, le=35.0),
) -> float:
    """Maximum affordable instalment."""
    buffer = discretionary_income * (100.0 - buffer_percentage) / 100.0
    residual_floor = 500.0 * (1.0 + float(dependants_count) * 0.2)
    floor_constraint = discretionary_income - residual_floor
    return max(0.0, min(max(0.0, buffer), max(0.0, floor_constraint)))


def affordability_verdict(
    proposed_instalment: float,
    max_affordable_instalment_unadjusted: float,
) -> float:
    """Verdict as numeric: 0=pass, 1=marginal, 2=fail."""
    if proposed_instalment <= max_affordable_instalment_unadjusted:
        return 0.0
    tolerance = max_affordable_instalment_unadjusted * 0.05
    if proposed_instalment <= (max_affordable_instalment_unadjusted + tolerance):
        return 1.0
    return 2.0


def discretionary_income_after(
    discretionary_income: float,
    proposed_instalment: float,
) -> float:
    """Remaining DI after instalment."""
    return discretionary_income - proposed_instalment


# Pipeline
pipeline = flow(
    gross_monthly_income,
    income_tax_monthly,
    unemployment_insurance_monthly,
    statutory_deductions,
    net_monthly_income,
    living_expenses_declared,
    living_expenses_norm,
    living_expenses,
    existing_obligations_external,
    existing_obligations_internal,
    existing_obligations,
    discretionary_income,
    max_affordable_instalment_unadjusted,
    affordability_verdict,
    discretionary_income_after,
).emit(
    "gross_monthly_income",
    "net_monthly_income",
    "living_expenses",
    "existing_obligations",
    "discretionary_income",
    "max_affordable_instalment_unadjusted",
    "affordability_verdict",
    "discretionary_income_after",
)
