"""
HTTP serving wrapper for Flex Loan assessment pipeline.

This demonstrates the shared credit core library in action - a Flex Loan
product using the shared income, affordability, and fees modules.

Usage:
    cd /home/sholto/Documents/Workspace/capitec/dsp-decision-engine/decider2/evaluation/00-shared-credit-core
    /path/to/.venv/bin/python -m decider2 serve serve_flex_loan.py --port 8101
"""

# Work around the namespace package issue
import sys
from pathlib import Path
_root_path = Path(__file__).parent.parent.parent.parent
if str(_root_path) not in sys.path:
    sys.path.insert(0, str(_root_path))

# Import directly from the graph and params modules to avoid the __init__.py issue
from decider2.graph import flow, module
from decider2.params import param
from pydantic import BaseModel, Field


# ==================================================================
# SHARED LIBRARY MODULES (normally imported from shared package)
# ==================================================================

def income_gross(
    declared_income: float | None,
    payslip_income: float | None,
    employment_type_code: int,
) -> float:
    """Determine gross monthly income from evidence waterfall."""
    if declared_income is not None and declared_income > 0:
        return declared_income
    if payslip_income is not None and payslip_income > 0:
        return payslip_income
    return 0.0


def income_source_code(
    declared_income: float | None,
    payslip_income: float | None,
) -> int:
    """Determine the evidence tier used."""
    if declared_income is not None and declared_income > 0:
        return 1
    if payslip_income is not None and payslip_income > 0:
        return 2
    return 0


def income_haircut(source_code: int) -> float:
    """Apply a verification haircut per evidence tier."""
    if source_code == 1:
        return 1.0
    elif source_code == 2:
        return 0.95
    else:
        return 0.9


def gross_monthly_income(
    income_gross: float, income_haircut: float
) -> float:
    """Verified gross monthly income after haircut applied."""
    return income_gross * income_haircut


def statutory_deductions(gross_monthly_income: float) -> float:
    """Statutory deductions: tax, unemployment insurance, retirement."""
    return gross_monthly_income * 0.18


def net_monthly_income(
    gross_monthly_income: float, statutory_deductions: float
) -> float:
    """Net income after statutory deductions."""
    return gross_monthly_income - statutory_deductions


class IncomeParams(BaseModel):
    tax_rate: float = Field(0.18, ge=0.0, le=1.0)


Income = module(
    income_gross,
    income_source_code,
    income_haircut,
    gross_monthly_income,
    statutory_deductions,
    net_monthly_income,
    name="income",
    params=IncomeParams,
)


# Affordability module
def living_expenses(
    declared_living_expenses: float,
    gross_monthly_income: float,
    dependants_count: int,
) -> float:
    """Applied living expenses after statutory minimum floor."""
    min_floor = 3000.0 + (500.0 * dependants_count)
    return max(declared_living_expenses, min_floor)


def existing_obligations(
    num_accounts: int,
    average_account_instalment: float,
) -> float:
    """Monthly cost of debt already held."""
    return num_accounts * average_account_instalment


def discretionary_income(
    net_monthly_income: float,
    living_expenses: float,
    existing_obligations: float,
) -> float:
    """Income remaining after living expenses and debt obligations."""
    return net_monthly_income - living_expenses - existing_obligations


def affordability_verdict_code(
    discretionary_income: float,
    instalment: float,
    params,
) -> int:
    """Pass/fail on affordability (1=pass, 2=marginal, 3=fail)."""
    if discretionary_income < 0:
        return 3
    if discretionary_income < instalment * params.marginal_threshold:
        return 2
    return 1


def max_affordable_instalment(
    discretionary_income: float,
    params,
) -> float:
    """Maximum monthly commitment after affordability buffer."""
    return discretionary_income * params.affordability_buffer


class AffordabilityParams(BaseModel):
    affordability_buffer: float = Field(0.35, ge=0.0, le=1.0)
    marginal_threshold: float = Field(0.5, ge=0.0, le=1.0)


Affordability = module(
    living_expenses,
    existing_obligations,
    discretionary_income,
    affordability_verdict_code,
    max_affordable_instalment,
    name="affordability",
    params=AffordabilityParams,
)


# Fees module
def initiation_fee(offered_amount: float, product_code: int, params) -> float:
    """One-off initiation fee, statutorily capped."""
    fee = offered_amount * params.initiation_fee_rate
    return min(fee, params.initiation_fee_cap)


def monthly_service_fee(offered_amount: float, params) -> float:
    """Monthly service fee, statutorily capped."""
    fee = offered_amount * params.monthly_service_fee_rate
    return min(fee, params.monthly_service_fee_cap)


def credit_life_premium(
    offered_amount: float,
    applicant_age_years: float,
    employment_type_code: int,
    params,
) -> float:
    """Monthly credit life insurance premium."""
    if applicant_age_years > params.max_age_for_cover:
        return 0.0
    base_rate = params.base_premium_rate
    if applicant_age_years > 60:
        base_rate *= params.senior_age_multiplier
    elif applicant_age_years < 25:
        base_rate *= params.youth_age_multiplier
    return (offered_amount / 1000.0) * base_rate


class FeesParams(BaseModel):
    initiation_fee_rate: float = Field(0.015, ge=0.0, le=0.1)
    initiation_fee_cap: float = Field(2000.0, ge=0.0)
    monthly_service_fee_rate: float = Field(0.005, ge=0.0, le=0.1)
    monthly_service_fee_cap: float = Field(150.0, ge=0.0)
    base_premium_rate: float = Field(0.5, ge=0.0)
    max_age_for_cover: float = Field(75.0, ge=0.0)
    youth_age_multiplier: float = Field(0.8, ge=0.0, le=2.0)
    senior_age_multiplier: float = Field(1.5, ge=0.0, le=2.0)


Fees = module(
    initiation_fee,
    monthly_service_fee,
    credit_life_premium,
    name="fees",
    params=FeesParams,
)


# ==================================================================
# FLEX LOAN PRODUCT-SPECIFIC LOGIC
# ==================================================================

class SharedParams(BaseModel):
    """Shared parameters across the pipeline."""
    prime_rate: float = Field(8.5, ge=0.0, le=20.0)
    product_margin: float = Field(2.5, ge=0.0, le=10.0)


def offered_amount(
    requested_amount: float,
    max_affordable_instalment: float,
    params,
) -> float:
    """Determine offered amount."""
    return min(requested_amount, params.product_maximum_amount)


def nominal_annual_rate(shared) -> float:
    """Determine offer rate."""
    return shared.prime_rate + shared.product_margin


def term_months(
    requested_term: int,
    params,
) -> int:
    """Determine loan term."""
    return min(requested_term, params.product_maximum_term)


def total_cost_of_credit(
    offered_amount: float,
    term_months: int,
    nominal_annual_rate: float,
    monthly_service_fee: float,
    initiation_fee: float,
) -> float:
    """Simplified total cost of credit calculation."""
    monthly_rate = nominal_annual_rate / 100.0 / 12.0
    interest = offered_amount * monthly_rate * term_months if monthly_rate else 0.0
    total_fees = initiation_fee + (monthly_service_fee * term_months)
    return offered_amount + interest + total_fees


class FlexLoanParams(BaseModel):
    product_maximum_amount: float = Field(500000.0, ge=0.0)
    product_minimum_amount: float = Field(2000.0, ge=0.0)
    product_maximum_term: int = Field(84, ge=1)
    product_minimum_term: int = Field(6, ge=1)


# Assemble the pipeline
TermDecision = module(term_months, name="term_decision", params=FlexLoanParams)
OfferedAmount = module(offered_amount, name="offered_amount", params=FlexLoanParams)
RateCard = module(nominal_annual_rate, name="rate_card")
CostCalculation = module(total_cost_of_credit, name="cost")

pipeline = (
    Income
    | Affordability
    | Fees
    | TermDecision
    | OfferedAmount
    | RateCard
    | CostCalculation
)

__all__ = ["pipeline", "SharedParams"]
