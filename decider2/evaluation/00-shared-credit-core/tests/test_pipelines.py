"""Integration tests for the shared credit core library pipelines."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.flex_loan import flex_loan, FlexLoanSharedParams, FlexLoanParams
from pipelines.access_facility import (
    access_facility,
    AccessFacilitySharedParams,
    AccessFacilityParams,
)
from modules.income import IncomeParams
from modules.affordability import AffordabilityParams
from modules.fees import FeesParams


class TestFlexLoanPipeline:
    """Test the Flex Loan product pipeline."""

    def test_flex_loan_end_to_end(self):
        """Test a complete Flex Loan assessment."""
        params = {
            "income": vars(IncomeParams()),
            "affordability": vars(AffordabilityParams()),
            "fees": vars(FeesParams()),
            "term_decision": vars(FlexLoanParams()),
            "offered_amount": vars(FlexLoanParams()),
        }
        shared = {
            "prime_rate": 8.5,
            "product_margin": 2.5,
        }

        result = flex_loan.score(
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
        assert "monthly_service_fee" in result
        assert "credit_life_premium" in result
        assert result["nominal_annual_rate"] == pytest.approx(11.0)  # 8.5 + 2.5

    def test_flex_loan_respects_product_limits(self):
        """Offered amount should respect product maximum."""
        params = {
            "income": vars(IncomeParams()),
            "affordability": vars(AffordabilityParams()),
            "fees": vars(FeesParams()),
            "term_decision": vars(FlexLoanParams()),
            "offered_amount": vars(FlexLoanParams()),
        }
        shared = {
            "prime_rate": 8.5,
            "product_margin": 2.5,
        }

        result = flex_loan.score(
            {
                "declared_income": 100000.0,
                "payslip_income": None,
                "employment_type_code": 1,
                "declared_living_expenses": 8000.0,
                "gross_monthly_income": 100000.0,
                "dependants_count": 0,
                "num_accounts": 0,
                "average_account_instalment": 0.0,
                "instalment": 5000.0,
                "offered_amount": 500000.0,
                "product_code": 10,
                "applicant_age_years": 35.0,
                "requested_amount": 1000000.0,  # Way above product max
                "requested_term": 100,
            },
            params=params,
            shared=shared,
        )

        # Product max is 500000
        assert result["offered_amount"] == pytest.approx(500000.0)


class TestAccessFacilityPipeline:
    """Test the Access Facility (revolving) pipeline."""

    def test_access_facility_end_to_end(self):
        """Test a complete Access Facility assessment."""
        params = {
            "income": vars(IncomeParams()),
            "affordability": vars(AffordabilityParams()),
            "fees": vars(FeesParams()),
            "limit_decision": vars(AccessFacilityParams()),
        }
        shared = {
            "prime_rate": 8.5,
            "product_margin": 3.0,
        }

        result = access_facility.score(
            {
                "declared_income": 45000.0,
                "payslip_income": None,
                "employment_type_code": 1,
                "declared_living_expenses": 6000.0,
                "gross_monthly_income": 45000.0,
                "dependants_count": 1,
                "num_accounts": 1,
                "average_account_instalment": 500.0,
                "instalment": 3000.0,
                "offered_amount": 100000.0,
                "product_code": 21,
                "applicant_age_years": 28.0,
            },
            params=params,
            shared=shared,
        )

        # Verify key outputs
        assert "approved_limit" in result
        assert "monthly_interest_rate" in result
        assert result["monthly_interest_rate"] == pytest.approx((8.5 + 3.0) / 100.0 / 12.0)

    def test_access_facility_respects_limit_maximum(self):
        """Approved limit should respect product maximum."""
        params = {
            "income": vars(IncomeParams()),
            "affordability": vars(AffordabilityParams()),
            "fees": vars(FeesParams()),
            "limit_decision": vars(AccessFacilityParams()),
        }
        shared = {
            "prime_rate": 8.5,
            "product_margin": 3.0,
        }

        result = access_facility.score(
            {
                "declared_income": 200000.0,
                "payslip_income": None,
                "employment_type_code": 1,
                "declared_living_expenses": 20000.0,
                "gross_monthly_income": 200000.0,
                "dependants_count": 0,
                "num_accounts": 0,
                "average_account_instalment": 0.0,
                "instalment": 5000.0,
                "offered_amount": 300000.0,
                "product_code": 21,
                "applicant_age_years": 40.0,
            },
            params=params,
            shared=shared,
        )

        # Product max is 300000
        # Approved limit should not exceed this
        assert result["approved_limit"] <= 300000.0


class TestLibraryReusability:
    """Test that modules can be reused across pipelines."""

    def test_income_module_reused(self):
        """Income module should produce consistent results across pipelines."""
        params_flex = {
            "income": vars(IncomeParams()),
            "affordability": vars(AffordabilityParams()),
            "fees": vars(FeesParams()),
            "term_decision": vars(FlexLoanParams()),
            "offered_amount": vars(FlexLoanParams()),
        }
        params_access = {
            "income": vars(IncomeParams()),
            "affordability": vars(AffordabilityParams()),
            "fees": vars(FeesParams()),
            "limit_decision": vars(AccessFacilityParams()),
        }
        shared = {"prime_rate": 8.5, "product_margin": 2.5}

        test_input = {
            "declared_income": 45000.0,
            "payslip_income": None,
            "employment_type_code": 1,
            "declared_living_expenses": 5000.0,
            "gross_monthly_income": 45000.0,
            "dependants_count": 1,
            "num_accounts": 1,
            "average_account_instalment": 500.0,
            "instalment": 4000.0,
            "offered_amount": 150000.0,
            "product_code": 10,
            "applicant_age_years": 30.0,
            "requested_amount": 150000.0,
            "requested_term": 60,
        }

        result_flex = flex_loan.score(test_input, params=params_flex, shared=shared)
        result_access = access_facility.score(test_input, params=params_access, shared=shared)

        # Both should produce the same net_monthly_income (from Income module)
        assert result_flex["net_monthly_income"] == pytest.approx(
            result_access["net_monthly_income"]
        )
