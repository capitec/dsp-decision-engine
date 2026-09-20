"""Income determination capability.

Core credit logic for establishing gross and net monthly income from various
evidence tiers, applying verification haircuts per tier.

Implements: Credit Policy §6.1
"""


def income_gross(
    declared_income: float | None,
    payslip_income: float | None,
    employment_type_code: int,
) -> float:
    """Determine gross monthly income from evidence waterfall.

    Uses declared income as tier 1, falls back to payslip-derived income.
    Implements: Credit Policy §6.1.1
    """
    if declared_income is not None and declared_income > 0:
        return declared_income
    if payslip_income is not None and payslip_income > 0:
        return payslip_income
    return 0.0


def income_source_code(
    declared_income: float | None,
    payslip_income: float | None,
) -> int:
    """Determine the evidence tier used to establish income.

    1 = declared, 2 = payslip, 0 = none established.
    """
    if declared_income is not None and declared_income > 0:
        return 1
    if payslip_income is not None and payslip_income > 0:
        return 2
    return 0


def income_haircut(source_code: int) -> float:
    """Apply a verification haircut per evidence tier.

    Tier 1 (declared): 0% haircut
    Tier 2 (payslip): 5% haircut
    Tier 3 (other): 10% haircut
    """
    if source_code == 1:
        return 1.0  # no haircut
    elif source_code == 2:
        return 0.95  # 5% haircut
    else:
        return 0.9  # 10% haircut


def gross_monthly_income(
    income_gross: float, income_haircut: float
) -> float:
    """Verified gross monthly income after haircut applied.

    Implements: Credit Policy §6.1.2
    """
    return income_gross * income_haircut


def statutory_deductions(gross_monthly_income: float) -> float:
    """Statutory deductions: tax, unemployment insurance, retirement.

    Simplified model: 18% of gross income.
    Implements: Credit Policy §6.2
    """
    return gross_monthly_income * 0.18


def net_monthly_income(
    gross_monthly_income: float, statutory_deductions: float
) -> float:
    """Net income after statutory deductions.

    Implements: Credit Policy §6.2.1
    """
    return gross_monthly_income - statutory_deductions
