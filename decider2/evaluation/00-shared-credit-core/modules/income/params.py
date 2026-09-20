"""Parameters for income capability."""

from pydantic import BaseModel, Field


class IncomeParams(BaseModel):
    """Income determination parameters.

    These are library-global parameters owned by Credit Systems and changed
    under a release process.
    """

    min_income_floor: float = Field(
        0.0, ge=0.0, description="Minimum income to consider established"
    )
    declared_haircut: float = Field(
        1.0, ge=0.0, le=1.0, description="Haircut on declared income"
    )
    payslip_haircut: float = Field(
        0.95, ge=0.0, le=1.0, description="Haircut on payslip income"
    )
    other_haircut: float = Field(
        0.9, ge=0.0, le=1.0, description="Haircut on other income"
    )
    tax_rate: float = Field(
        0.18, ge=0.0, le=1.0, description="Statutory tax and deductions rate"
    )
