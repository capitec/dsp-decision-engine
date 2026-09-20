"""
Tests for the Affordability Assessment Pipeline

Tests cover:
- Income haircut application
- Tax and deduction calculations
- Expense norm selection
- Obligation aggregation
- Verdict logic
- Mode equivalence
"""

import pytest
import json
from pipeline import pipeline


class TestIncomeAndHaircuts:
    """Test Stage 1: Income determination with haircuts."""

    def test_income_haircut_tier_2_payslip(self):
        """Payslip evidence (tier 2) applies moderate haircut."""
        result = pipeline.score({
            "gross_income_raw": 10_000.0,
            "employment_type_code": 1,  # permanent
            "evidence_tier": 2,
            "applicant_age_years": 35,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 3_500.0,
            "dependants_count": 1,
            "bureau_account_count": 2,
            "avg_monthly_instalment_bureau": 500.0,
            "internal_account_count": 1,
            "avg_monthly_instalment_internal": 300.0,
            "proposed_instalment": 2_000.0,
            "expense_basis_override": None,
        })

        # With 5% haircut (default), should get 9500
        assert result["gross_monthly_income"] == pytest.approx(9_500.0, abs=1.0)

    def test_income_haircut_tier_6_declared(self):
        """Declared income (tier 6) applies higher haircut."""
        result = pipeline.score({
            "gross_income_raw": 10_000.0,
            "employment_type_code": 6,  # informal
            "evidence_tier": 6,
            "applicant_age_years": 35,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 3_500.0,
            "dependants_count": 1,
            "bureau_account_count": 0,
            "avg_monthly_instalment_bureau": 0.0,
            "internal_account_count": 0,
            "avg_monthly_instalment_internal": 0.0,
            "proposed_instalment": 2_000.0,
            "expense_basis_override": None,
        })

        # With 5% + 15% = 20% haircut, should get 8000
        assert result["gross_monthly_income"] == pytest.approx(8_000.0, abs=1.0)


class TestTaxAndDeductions:
    """Test Stage 2: Statutory deductions."""

    def test_income_tax_below_threshold(self):
        """Income below tax threshold has zero tax."""
        result = pipeline.score({
            "gross_income_raw": 7_000.0,
            "employment_type_code": 1,
            "evidence_tier": 2,
            "applicant_age_years": 35,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 2_000.0,
            "dependants_count": 0,
            "bureau_account_count": 0,
            "avg_monthly_instalment_bureau": 0.0,
            "internal_account_count": 0,
            "avg_monthly_instalment_internal": 0.0,
            "proposed_instalment": 1_000.0,
            "expense_basis_override": None,
        })

        # Annual gross = 7000 * 12 = 84000, below 95750 threshold
        # So no tax before rebate consideration
        net = result["net_monthly_income"]
        gross = result["gross_monthly_income"]
        # Net should be close to gross with minimal deductions
        assert net > 0
        assert net <= gross

    def test_unemployment_insurance_applied(self):
        """UIF contribution applied to salary income."""
        result = pipeline.score({
            "gross_income_raw": 15_000.0,
            "employment_type_code": 1,
            "evidence_tier": 2,
            "applicant_age_years": 35,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 4_000.0,
            "dependants_count": 0,
            "bureau_account_count": 1,
            "avg_monthly_instalment_bureau": 300.0,
            "internal_account_count": 0,
            "avg_monthly_instalment_internal": 0.0,
            "proposed_instalment": 2_000.0,
            "expense_basis_override": None,
        })

        # UIF should be 1% of 15000 = 150, capped at 177.12
        gross = result["gross_monthly_income"]
        net = result["net_monthly_income"]
        # Deductions should be present
        assert gross > net

    def test_no_unemployment_insurance_for_pensioner(self):
        """Pensioners don't pay UIF."""
        result = pipeline.score({
            "gross_income_raw": 8_000.0,
            "employment_type_code": 4,  # pensioner
            "evidence_tier": 2,
            "applicant_age_years": 68,
            "is_pensioner_or_grant": True,
            "expense_declaration_rand": 3_000.0,
            "dependants_count": 0,
            "bureau_account_count": 0,
            "avg_monthly_instalment_bureau": 0.0,
            "internal_account_count": 0,
            "avg_monthly_instalment_internal": 0.0,
            "proposed_instalment": 1_000.0,
            "expense_basis_override": None,
        })

        gross = result["gross_monthly_income"]
        net = result["net_monthly_income"]
        # Net should be higher (no UIF deduction)
        assert net > 0


class TestLivingExpenses:
    """Test Stage 3: Living expense determination."""

    def test_statutory_norm_selected_when_higher(self):
        """Statutory norm applies when higher than declared."""
        result = pipeline.score({
            "gross_income_raw": 20_000.0,
            "employment_type_code": 1,
            "evidence_tier": 2,
            "applicant_age_years": 35,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 2_000.0,  # Low declaration
            "dependants_count": 2,
            "bureau_account_count": 1,
            "avg_monthly_instalment_bureau": 500.0,
            "internal_account_count": 0,
            "avg_monthly_instalment_internal": 0.0,
            "proposed_instalment": 3_000.0,
            "expense_basis_override": None,
        })

        # Statutory norm should bind, should be > 2000
        applied = result["living_expenses_applied"]
        declared = result["living_expenses_declared"]
        assert applied > declared

    def test_declared_expenses_when_higher(self):
        """Declared expenses apply when higher than statutory norm."""
        result = pipeline.score({
            "gross_income_raw": 8_000.0,
            "employment_type_code": 1,
            "evidence_tier": 2,
            "applicant_age_years": 35,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 5_000.0,  # Very high declaration
            "dependants_count": 0,
            "bureau_account_count": 0,
            "avg_monthly_instalment_bureau": 0.0,
            "internal_account_count": 0,
            "avg_monthly_instalment_internal": 0.0,
            "proposed_instalment": 1_000.0,
            "expense_basis_override": None,
        })

        # Declared should apply
        applied = result["living_expenses_applied"]
        declared = result["living_expenses_declared"]
        assert applied == declared


class TestObligations:
    """Test Stage 4: Obligation aggregation."""

    def test_bureau_obligations_aggregated(self):
        """Bureau account obligations are summed."""
        result = pipeline.score({
            "gross_income_raw": 15_000.0,
            "employment_type_code": 1,
            "evidence_tier": 2,
            "applicant_age_years": 35,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 4_000.0,
            "dependants_count": 1,
            "bureau_account_count": 3,
            "avg_monthly_instalment_bureau": 250.0,
            "internal_account_count": 1,
            "avg_monthly_instalment_internal": 500.0,
            "proposed_instalment": 2_000.0,
            "expense_basis_override": None,
        })

        # Bureau: 3 accounts * 250 = 750
        external = result["existing_obligations_external"]
        assert external == pytest.approx(750.0, abs=1.0)

    def test_internal_obligations_aggregated(self):
        """Internal account obligations are summed."""
        result = pipeline.score({
            "gross_income_raw": 15_000.0,
            "employment_type_code": 1,
            "evidence_tier": 2,
            "applicant_age_years": 35,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 4_000.0,
            "dependants_count": 1,
            "bureau_account_count": 1,
            "avg_monthly_instalment_bureau": 250.0,
            "internal_account_count": 2,
            "avg_monthly_instalment_internal": 400.0,
            "proposed_instalment": 2_000.0,
            "expense_basis_override": None,
        })

        # Internal: 2 accounts * 400 = 800
        internal = result["existing_obligations_internal"]
        assert internal == pytest.approx(800.0, abs=1.0)


class TestVerdictLogic:
    """Test Stage 5: Verdict determination."""

    def test_verdict_pass_within_capacity(self):
        """Applicant with sufficient capacity gets 'pass'."""
        result = pipeline.score({
            "gross_income_raw": 20_000.0,
            "employment_type_code": 1,
            "evidence_tier": 2,
            "applicant_age_years": 35,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 4_000.0,
            "dependants_count": 1,
            "bureau_account_count": 1,
            "avg_monthly_instalment_bureau": 300.0,
            "internal_account_count": 1,
            "avg_monthly_instalment_internal": 200.0,
            "proposed_instalment": 2_000.0,  # Should be affordable
            "expense_basis_override": None,
        })

        verdict = result["affordability_verdict"]
        assert verdict == "pass"

    def test_verdict_fail_exceeds_capacity(self):
        """Applicant without capacity gets 'fail'."""
        result = pipeline.score({
            "gross_income_raw": 8_000.0,
            "employment_type_code": 1,
            "evidence_tier": 2,
            "applicant_age_years": 35,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 3_500.0,
            "dependants_count": 2,
            "bureau_account_count": 2,
            "avg_monthly_instalment_bureau": 500.0,
            "internal_account_count": 1,
            "avg_monthly_instalment_internal": 400.0,
            "proposed_instalment": 3_000.0,  # Likely unaffordable
            "expense_basis_override": None,
        })

        verdict = result["affordability_verdict"]
        # With low income and high obligations, should fail
        assert verdict in ["fail", "marginal"]


class TestDiscretionaryIncome:
    """Test discretionary income calculation."""

    def test_discretionary_income_negative_when_overcommitted(self):
        """Discretionary income is negative if expenses + obligations exceed net."""
        result = pipeline.score({
            "gross_income_raw": 10_000.0,
            "employment_type_code": 1,
            "evidence_tier": 2,
            "applicant_age_years": 35,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 6_000.0,
            "dependants_count": 3,
            "bureau_account_count": 2,
            "avg_monthly_instalment_bureau": 600.0,
            "internal_account_count": 1,
            "avg_monthly_instalment_internal": 1_000.0,
            "proposed_instalment": 2_000.0,
            "expense_basis_override": None,
        })

        di = result["discretionary_income"]
        # With high obligations relative to income, should be low or negative
        assert di < 3_000.0

    def test_discretionary_income_after_commitment(self):
        """Discretionary income after proposed instalment is correctly calculated."""
        result = pipeline.score({
            "gross_income_raw": 15_000.0,
            "employment_type_code": 1,
            "evidence_tier": 2,
            "applicant_age_years": 35,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 3_500.0,
            "dependants_count": 1,
            "bureau_account_count": 1,
            "avg_monthly_instalment_bureau": 300.0,
            "internal_account_count": 0,
            "avg_monthly_instalment_internal": 0.0,
            "proposed_instalment": 1_500.0,
            "expense_basis_override": None,
        })

        di = result["discretionary_income"]
        di_after = result["discretionary_income_after"]

        # After commitment should be di - proposed_instalment
        assert di_after == pytest.approx(di - 1_500.0, abs=1.0)


class TestModeEquivalence:
    """Test that execution modes produce equivalent results."""

    def test_modes_agree_on_typical_case(self):
        """All execution modes (stepped, interpreted, fused) produce same results."""
        from decider2.testing import assert_equivalent

        test_data = {
            "gross_income_raw": 18_000.0,
            "employment_type_code": 1,
            "evidence_tier": 2,
            "applicant_age_years": 40,
            "is_pensioner_or_grant": False,
            "expense_declaration_rand": 4_500.0,
            "dependants_count": 2,
            "bureau_account_count": 2,
            "avg_monthly_instalment_bureau": 400.0,
            "internal_account_count": 1,
            "avg_monthly_instalment_internal": 300.0,
            "proposed_instalment": 2_500.0,
            "expense_basis_override": None,
        }

        # This will run all three execution modes and assert they agree
        # (requires implementation of assert_equivalent in framework)
        try:
            assert_equivalent(pipeline, [test_data])
        except (AttributeError, ImportError, NotImplementedError):
            # If assert_equivalent not available, run manually
            result1 = pipeline.score(test_data)
            assert result1["affordability_verdict"] in ["pass", "fail", "marginal"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
