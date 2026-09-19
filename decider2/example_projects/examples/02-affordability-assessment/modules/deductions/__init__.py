"""Stage 3 assembly."""

from decider2 import module, step

from modules.deductions.statutory import (
    EmployerUIF,
    court_order_credit_agreement_keys,
    court_ordered_deductions_cents,
    payslip_decomposition_failed,
    retirement_basis,
    retirement_cents,
    retirement_imputed_cents,
    retirement_observed_cents,
    statutory_deductions_cents,
    undecomposed_payslip_deductions_cents,
)
from modules.deductions.tax import (
    annual_gross_cents,
    income_tax_cents,
    paye_bracket_index,
    rebate_class,
)


@step(
    output="net_monthly_income_cents",
    description="Gross monthly income less statutory deductions. Rung two of the prescribed ladder.",
)
def net_monthly_income_cents(
    gross_monthly_income_cents: int,
    statutory_deductions_cents: int,
) -> int:
    pass  # gross - statutory


Deductions = module(
    rebate_class,
    annual_gross_cents,
    paye_bracket_index,
    income_tax_cents,
    EmployerUIF,
    retirement_observed_cents,
    retirement_imputed_cents,
    retirement_cents,
    retirement_basis,
    statutory_deductions_cents,
    undecomposed_payslip_deductions_cents,
    payslip_decomposition_failed,
    court_ordered_deductions_cents,
    court_order_credit_agreement_keys,
    net_monthly_income_cents,
    name="deductions",
    contract="contracts/deductions.json",
    taps=["income_tax_cents", "uif_cents", "retirement_cents", "retirement_basis"],
)

# Acceptance criterion 2: "a change to the tax calculation reaches all four
# modes at once, demonstrably". The demonstration is not a code review. It is
# that `Deductions` appears exactly once in pipelines/affordability.py, no
# profile in policy/modes.py binds anything in its namespace, and
# `tests/test_properties.py::test_all_modes_share_one_kernel` fails if a mode
# ever acquires its own copy.
