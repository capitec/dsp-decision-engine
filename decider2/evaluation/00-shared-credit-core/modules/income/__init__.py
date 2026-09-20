"""Income determination — establishing gross and net monthly income.

Published interface:
- Inputs: declared_income, payslip_income, employment_type_code
- Outputs: gross_monthly_income, net_monthly_income, income_source_code
- Params: IncomeParams
"""

import sys
from pathlib import Path

# Ensure decider2 can be imported from the framework location
root = Path(__file__).parent.parent.parent.parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from decider2 import module

from . import steps
from .params import IncomeParams

Income = module(
    steps.income_gross,
    steps.income_source_code,
    steps.income_haircut,
    steps.gross_monthly_income,
    steps.statutory_deductions,
    steps.net_monthly_income,
    name="income",
    params=IncomeParams,
)

__all__ = ["Income", "IncomeParams"]
