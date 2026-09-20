"""
Tests for the shared credit core library.

Covers:
- Income determination with evidence tiers
- Affordability assessment
- Fee calculations
- Pipeline integration
- Mode equivalence testing
"""

import pytest
import sys
from pathlib import Path

# Fix imports to work around namespace package issue
_root_path = Path(__file__).parent.parent.parent.parent
if str(_root_path) not in sys.path:
    sys.path.insert(0, str(_root_path))

from pipeline import (
    Income, Affordability, Fees,
    IncomeParams, AffordabilityParams, FeesParams,
    pipeline, SharedParams, FlexLoanParams,
)


class TestIncomeModule:
    """Test income determination capability."""

    def test_declared_income_no_haircut(self):
        """Declared income (tier 1) should not have haircut."""
        params = {"income": {}}
        result = Income.score(
            {
                "declared_income": 45000.0,
                "payslip_income": None,
                "employment_type_code": 1,
            },
            params=params,
        )
        assert result["gross_monthly_income"] == pytest.approx(45000.0)
        assert result["income_source_code"] == 1
        assert result["net_monthly_income"] == pytest.approx(45000.0 * 0.82)

    def test_payslip_income_with_haircut(self):
        """Payslip income (tier 2) should have 5% haircut."""
        params = {"income": {}}
        result = Income.score(
            {
                "declared_income": None,
                "payslip_income": 50000.0,
                "employment_type_code": 1,
            },
            params=params,
        )
        # Payslip 50000, haircut 0.95 → 47500 * 0.82
        assert result["gross_monthly_income"] == pytest.approx(50000.0 * 0.95)
        assert result["income_source_code"] == 2
        assert result["net_monthly_income"] == pytest.approx(50000.0 * 0.95 * 0.82)

    def test_no_income(self):
        """When no income sources exist."""
        params = {"income": {}}
        result = Income.score(
            {
                "declared_income": None,
                "payslip_income": None,
                "employment_type_code": 1,
            },
            params=params,
        )
        assert result["gross_monthly_income"] == 0.0
        assert result["income_source_code"] == 0
        assert result["net_monthly_income"] == 0.0


class TestAffordabilityModule:
    """Test affordability assessment capability."""

    def test_living_expenses_floor(self):
        """Living expenses should not go below statutory floor."""
        params = {"affordability": {}}
        result = Affordability.score(
            {
                "declared_living_expenses": 1000.0,
                "gross_monthly_income": 50000.0,
                "dependants_count": 0,
                "num_accounts": 0,
                "average_account_instalment": 0.0,
                "instalment": 5000.0,
            },
            params=params,
        )
        # Floor = 3000 + (500 * 0) = 3000
        assert result["living_expenses"] == pytest.approx(3000.0)

    def test_living_expenses_above_floor(self):
        """When declared expenses are above floor, use declared."""
        params = {"affordability": {}}
        result = Affordability.score(
            {
                "declared_living_expenses": 8000.0,
                "gross_monthly_income": 50000.0,
                "dependants_count": 0,
                "num_accounts": 0,
                "average_account_instalment": 0.0,
                "instalment": 5000.0,
            },
            params=params,
        )
        assert result["living_expenses"] == pytest.approx(8000.0)

    def test_discretionary_income(self):
        """Discretionary income = net - expenses - obligations."""
        params = {"affordability": {}}
        result = Affordability.score(
            {
                "declared_living_expenses": 5000.0,
                "gross_monthly_income": 45000.0,
                "dependants_count": 0,
                "num_accounts": 3,
                "average_account_instalment": 1000.0,
                "instalment": 5000.0,
                "net_monthly_income": 40000.0,
            },
            params=params,
        )
        # Discretionary = 40000 - 5000 - 3000 = 32000
        assert result["discretionary_income"] == pytest.approx(32000.0)

    def test_max_affordable_instalment(self):
        """Max affordable instalment applies buffer to discretionary income."""
        params = {"affordability": {}}
        result = Affordability.score(
            {
                "declared_living_expenses": 5000.0,
                "gross_monthly_income": 50000.0,
                "dependants_count": 0,
                "num_accounts": 0,
                "average_account_instalment": 0.0,
                "instalment": 5000.0,
                "net_monthly_income": 41000.0,
            },
            params=params,
        )
        # Discretionary = 41000 - 5000 - 0 = 36000
        # Max affordable = 36000 * 0.35 = 12600
        assert result["max_affordable_instalment"] == pytest.approx(12600.0)


class TestFeesModule:
    """Test fee calculation capability."""

    def test_initiation_fee_capped(self):
        """Initiation fee should be capped at statutory maximum."""
        params = {"fees": {}}
        result = Fees.score(
            {
                "offered_amount": 500000.0,
                "product_code": 10,
                "applicant_age_years": 35.0,
                "employment_type_code": 1,
            },
            params=params,
        )
        # Fee = 500000 * 0.015 = 7500, capped at 2000
        assert result["initiation_fee"] == pytest.approx(2000.0)

    def test_credit_life_premium_by_age(self):
        """Credit life premium should adjust by age."""
        params_young = {"fees": {}}
        params_normal = {"fees": {}}

        # Young applicant (< 25)
        result_young = Fees.score(
            {
                "offered_amount": 100000.0,
                "product_code": 10,
                "applicant_age_years": 22.0,
                "employment_type_code": 1,
            },
            params=params_young,
        )

        # Normal applicant
        result_normal = Fees.score(
            {
                "offered_amount": 100000.0,
                "product_code": 10,
                "applicant_age_years": 35.0,
                "employment_type_code": 1,
            },
            params=params_normal,
        )

        # Young should be cheaper (0.8x multiplier)
        assert result_young["credit_life_premium"] < result_normal["credit_life_premium"]


class TestFlexLoanPipeline:
    """Test complete Flex Loan assessment pipeline."""

    def test_end_to_end_assessment(self):
        """Test a complete application assessment."""
        params = {
            "income": {},
            "affordability": {},
            "fees": {},
            "term_decision": {},
            "offered_amount": {},
        }
        shared = {
            "prime_rate": 8.5,
            "product_margin": 2.5,
        }

        result = pipeline.score(
            {
                "declared_income": 50000.0,
                "payslip_income": None,
                "employment_type_code": 1,
                "declared_living_expenses": 8000.0,
                "gross_monthly_income": 50000.0,
                "dependants_count": 1,
                "num_accounts": 2,
                "average_account_instalment": 1500.0,
                "instalment": 5000.0,
                "offered_amount": 200000.0,
                "product_code": 10,
                "applicant_age_years": 35.0,
                "requested_amount": 250000.0,
                "requested_term": 60,
            },
            params=params,
            shared=shared,
        )

        # Verify key outputs are produced
        assert "net_monthly_income" in result
        assert "discretionary_income" in result
        assert "affordability_verdict_code" in result
        assert "initiation_fee" in result
        assert result["nominal_annual_rate"] == pytest.approx(11.0)

    def test_high_earner(self):
        """Test assessment of high-earning applicant."""
        params = {
            "income": {},
            "affordability": {},
            "fees": {},
            "term_decision": {},
            "offered_amount": {},
        }
        shared = {
            "prime_rate": 8.5,
            "product_margin": 2.5,
        }

        result = pipeline.score(
            {
                "declared_income": 150000.0,
                "payslip_income": None,
                "employment_type_code": 1,
                "declared_living_expenses": 15000.0,
                "gross_monthly_income": 150000.0,
                "dependants_count": 0,
                "num_accounts": 1,
                "average_account_instalment": 2000.0,
                "instalment": 10000.0,
                "offered_amount": 450000.0,
                "product_code": 10,
                "applicant_age_years": 40.0,
                "requested_amount": 500000.0,
                "requested_term": 72,
            },
            params=params,
            shared=shared,
        )

        assert result["discretionary_income"] > 100000.0
        assert result["affordability_verdict_code"] == 1  # pass


class TestPipelineEquivalence:
    """Test that modes agree across execution paths."""

    def test_modes_agree_simple_case(self):
        """Test equivalence across interpreted and stepped modes."""
        from decider2.testing import assert_equivalent

        params = {
            "income": {},
            "affordability": {},
            "fees": {},
            "term_decision": {},
            "offered_amount": {},
        }
        shared = {
            "prime_rate": 8.5,
            "product_margin": 2.5,
        }

        test_record = {
            "declared_income": 60000.0,
            "payslip_income": None,
            "employment_type_code": 1,
            "declared_living_expenses": 6000.0,
            "gross_monthly_income": 60000.0,
            "dependants_count": 1,
            "num_accounts": 2,
            "average_account_instalment": 800.0,
            "instalment": 3500.0,
            "offered_amount": 180000.0,
            "product_code": 10,
            "applicant_age_years": 32.0,
            "requested_amount": 200000.0,
            "requested_term": 60,
        }

        assert_equivalent(
            pipeline,
            test_record,
            params=params,
            shared=shared,
        )
