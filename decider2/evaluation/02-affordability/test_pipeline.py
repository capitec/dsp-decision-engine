"""
Tests for the Affordability Assessment Pipeline
"""

import pytest
from pipeline import pipeline


class TestBasicFunctionality:
    """Test basic pipeline functionality."""

    def test_pipeline_runs(self):
        """Pipeline executes without errors."""
        result = pipeline.score({
            'gross_income_raw': 10_000.0,
            'employment_type_code': 1,
            'evidence_tier': 2,
            'applicant_age_years': 35,
            'is_pensioner_or_grant': False,
            'expense_declaration_rand': 3_500.0,
            'dependants_count': 1,
            'bureau_account_count': 2,
            'avg_monthly_instalment_bureau': 500.0,
            'internal_account_count': 1,
            'avg_monthly_instalment_internal': 300.0,
            'proposed_instalment': 2_000.0,
            'court_ordered_deductions': 0.0,
        })
        assert result['gross_monthly_income'] > 0
        assert result['net_monthly_income'] > 0
        assert result['affordability_verdict'] in [0.0, 1.0, 2.0]

    def test_income_haircut_applied(self):
        """Income haircut reduces gross income."""
        result = pipeline.score({
            'gross_income_raw': 10_000.0,
            'employment_type_code': 1,
            'evidence_tier': 2,  # 5% haircut
            'applicant_age_years': 35,
            'is_pensioner_or_grant': False,
            'expense_declaration_rand': 3_000.0,
            'dependants_count': 0,
            'bureau_account_count': 0,
            'avg_monthly_instalment_bureau': 0.0,
            'internal_account_count': 0,
            'avg_monthly_instalment_internal': 0.0,
            'proposed_instalment': 1_000.0,
            'court_ordered_deductions': 0.0,
        })
        # 5% of 10000 = 500, so expect 9500
        assert result['gross_monthly_income'] == pytest.approx(9_500.0, abs=1.0)

    def test_verdict_pass_within_capacity(self):
        """Applicant with sufficient capacity passes."""
        result = pipeline.score({
            'gross_income_raw': 30_000.0,
            'employment_type_code': 1,
            'evidence_tier': 2,
            'applicant_age_years': 35,
            'is_pensioner_or_grant': False,
            'expense_declaration_rand': 5_000.0,
            'dependants_count': 0,
            'bureau_account_count': 1,
            'avg_monthly_instalment_bureau': 500.0,
            'internal_account_count': 0,
            'avg_monthly_instalment_internal': 0.0,
            'proposed_instalment': 5_000.0,
            'court_ordered_deductions': 0.0,
        })
        assert result['affordability_verdict'] == 0.0  # pass

    def test_verdict_fail_exceeds_capacity(self):
        """Applicant without capacity fails."""
        result = pipeline.score({
            'gross_income_raw': 5_000.0,
            'employment_type_code': 1,
            'evidence_tier': 2,
            'applicant_age_years': 35,
            'is_pensioner_or_grant': False,
            'expense_declaration_rand': 3_000.0,
            'dependants_count': 3,
            'bureau_account_count': 3,
            'avg_monthly_instalment_bureau': 500.0,
            'internal_account_count': 1,
            'avg_monthly_instalment_internal': 300.0,
            'proposed_instalment': 3_000.0,
            'court_ordered_deductions': 0.0,
        })
        assert result['affordability_verdict'] == 2.0  # fail

    def test_discretionary_income_calculation(self):
        """Discretionary income is correctly calculated."""
        result = pipeline.score({
            'gross_income_raw': 20_000.0,
            'employment_type_code': 1,
            'evidence_tier': 2,
            'applicant_age_years': 35,
            'is_pensioner_or_grant': False,
            'expense_declaration_rand': 4_000.0,
            'dependants_count': 1,
            'bureau_account_count': 1,
            'avg_monthly_instalment_bureau': 400.0,
            'internal_account_count': 0,
            'avg_monthly_instalment_internal': 0.0,
            'proposed_instalment': 2_000.0,
            'court_ordered_deductions': 0.0,
        })
        # DI = net - expenses - obligations
        di = result['discretionary_income']
        di_after = result['discretionary_income_after']
        # DI after should be DI - proposed instalment
        assert di_after == pytest.approx(di - 2_000.0, abs=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
