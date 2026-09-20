"""Tests for income determination capability."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.income import Income, IncomeParams


class TestIncomeModule:
    """Test the income module's income determination logic."""

    def test_income_from_declared(self):
        """Declared income (tier 1) has no haircut."""
        params = IncomeParams()
        result = Income.score(
            {
                "declared_income": 45000.0,
                "payslip_income": None,
                "employment_type_code": 1,
            },
            params={"income": vars(params)},
        )
        # Declared 45000, haircut 1.0, deductions 18% → net = 45000 * 1.0 * 0.82
        assert result["gross_monthly_income"] == pytest.approx(45000.0)
        assert result["income_source_code"] == 1
        assert result["net_monthly_income"] == pytest.approx(45000.0 * 0.82)

    def test_income_from_payslip(self):
        """Payslip income (tier 2) has 5% haircut."""
        params = IncomeParams()
        result = Income.score(
            {
                "declared_income": None,
                "payslip_income": 50000.0,
                "employment_type_code": 1,
            },
            params={"income": vars(params)},
        )
        # Payslip 50000, haircut 0.95 → 47500 * 0.82 = 38950
        assert result["gross_monthly_income"] == pytest.approx(50000.0 * 0.95)
        assert result["income_source_code"] == 2
        assert result["net_monthly_income"] == pytest.approx(50000.0 * 0.95 * 0.82)

    def test_no_income(self):
        """When no income sources exist."""
        params = IncomeParams()
        result = Income.score(
            {
                "declared_income": None,
                "payslip_income": None,
                "employment_type_code": 1,
            },
            params={"income": vars(params)},
        )
        assert result["gross_monthly_income"] == 0.0
        assert result["income_source_code"] == 0
        assert result["net_monthly_income"] == 0.0

    def test_equivalence_across_modes(self):
        """Test that interpreted, stepped, and fused modes agree."""
        from decider2.testing import assert_equivalent

        params = IncomeParams()
        test_cases = [
            {
                "declared_income": 45000.0,
                "payslip_income": None,
                "employment_type_code": 1,
            },
            {
                "declared_income": None,
                "payslip_income": 50000.0,
                "employment_type_code": 1,
            },
            {
                "declared_income": 0.0,
                "payslip_income": 0.0,
                "employment_type_code": 1,
            },
        ]

        for record in test_cases:
            assert_equivalent(
                Income,
                record,
                params={"income": vars(params)},
            )
