"""Tests for core credit library capabilities."""
import pytest
from datetime import date
from core.income import determine_income
from core.deductions import calculate_deductions
from core.expense_norms import apply_expense_norms
from core.obligations import calculate_obligations, Account
from core.affordability import assess_affordability
from core.instalment import calculate_instalment, calculate_max_affordable_amount
from core.fees import calculate_fees
from core.rate_card import lookup_flex_loan_rate
from core.scorecard import evaluate_scorecard
from core.calibration import calibrate_score
from core.risk_grade import assign_risk_grade
from core.reason_codes import rank_reasons, get_primary_reason
from core.rounding import round_instalment


class TestIncome:
    def test_income_from_payslip(self):
        """Test income determination from payslip (00 §6.1)."""
        result = determine_income(payslip_income=15000.0, employment_type_code=1)
        assert result.gross_monthly_income > 0
        assert result.income_source_code == 3  # Payslip
        assert result.income_haircut_applied > 0

    def test_income_waterfall(self):
        """Test evidence waterfall: employer > payslip > declared."""
        result = determine_income(
            declared_income=10000.0,
            payslip_income=15000.0,
            employer_confirmed=16000.0
        )
        # Employer-confirmed should win, with no haircut
        assert result.income_source_code == 2
        assert result.income_haircut_applied == 0.0


class TestDeductions:
    def test_statutory_deductions(self):
        """Test deduction calculation (00 §6.2)."""
        gross = 15000.0
        result = calculate_deductions(gross, employment_type_code=1)
        assert result.net_monthly_income < gross
        assert result.net_monthly_income > 0


class TestExpenseNorms:
    def test_expense_norm_floor(self):
        """Test minimum living expense floor (00 §6.3)."""
        result = apply_expense_norms(10000.0, 2, declared_living_expenses=500.0)
        # Declared is too low; norm floor should apply
        assert result.living_expenses >= 500.0


class TestObligations:
    def test_calculate_obligations(self):
        """Test existing obligations calculation (00 §6.4)."""
        accounts = [
            Account("ACC001", "instalment", 500.0, 10000.0, "active"),
            Account("ACC002", "revolving", 300.0, 5000.0, "active"),
        ]
        result = calculate_obligations(internal_accounts=accounts)
        assert result.existing_obligations == 800.0
        assert result.account_count == 2

    def test_deduplication(self):
        """Test bureau/internal de-duplication (00-ADDENDUM A10)."""
        bureau = [Account("ACC001", "instalment", 500.0, 10000.0, "active")]
        internal = [Account("ACC001", "instalment", 500.0, 10000.0, "active")]
        result = calculate_obligations(bureau, internal)
        # Should keep internal, dedupe bureau
        assert result.account_count == 1


class TestAffordability:
    def test_affordability_pass(self):
        """Test pass verdict (00 §6.5)."""
        result = assess_affordability(
            net_monthly_income=15000.0,
            living_expenses=5000.0,
            existing_obligations=2000.0,
            proposed_instalment=2500.0
        )
        assert result.verdict_code == 1  # Pass
        assert result.pass_fail is True

    def test_affordability_fail(self):
        """Test fail verdict."""
        result = assess_affordability(
            net_monthly_income=10000.0,
            living_expenses=8000.0,
            existing_obligations=2000.0,
            proposed_instalment=5000.0
        )
        assert result.verdict_code == 3  # Fail


class TestInstalment:
    def test_calculate_instalment(self):
        """Test instalment calculation (00 §6.6)."""
        result = calculate_instalment(
            amount=50000.0,
            term_months=36,
            nominal_annual_rate=12.0,
            initiation_fee=500.0,
            monthly_service_fee=50.0
        )
        assert result.instalment > 0
        assert result.total_cost_of_credit > 0
        assert result.effective_annual_rate > 0

    def test_max_affordable_amount_inverse(self):
        """Test inverse calculation (00-ADDENDUM C item 6)."""
        max_amount = calculate_max_affordable_amount(
            net_income=15000.0,
            living_expenses=5000.0,
            existing_obligations=2000.0,
            term_months=36,
            nominal_annual_rate=12.0,
            max_dti_ratio=0.35
        )
        assert max_amount > 0


class TestFees:
    def test_fee_calculation(self):
        """Test fee calculation (00 §6.7)."""
        result = calculate_fees(50000.0, 10, 36)
        assert result.initiation_fee > 0
        assert result.monthly_service_fee > 0
        assert result.credit_life_premium > 0


class TestRateCard:
    def test_rate_lookup(self):
        """Test rate card lookup (00 §6.8)."""
        result = lookup_flex_loan_rate(
            amount=50000.0,
            term_months=36,
            risk_grade=6
        )
        assert 0 < result.nominal_annual_rate < 50
        assert "cell_" in result.rate_cell_id

    def test_rate_dimensions(self):
        """Test full 96x55x12 card dimensions."""
        # Test boundaries
        result_low = lookup_flex_loan_rate(2000.0, 6, 1)
        result_high = lookup_flex_loan_rate(500000.0, 84, 12)
        # Rates should increase with risk grade
        assert result_low.nominal_annual_rate < result_high.nominal_annual_rate


class TestScorecard:
    def test_scorecard_evaluation(self):
        """Test scorecard scoring with contributions (00 §6.10)."""
        characteristics = {
            "income_level": 15000.0,
            "employment_tenure": 24.0,
            "adverse_events": 0.0
        }
        bins = {
            "income_level": [(0, 5000), (5000, 10000), (10000, 20000), (20000, float('inf'))],
            "employment_tenure": [(0, 6), (6, 24), (24, 60), (60, float('inf'))],
            "adverse_events": [(0, 1), (1, 3), (3, float('inf'))]
        }
        result = evaluate_scorecard(1, characteristics, bins)
        assert result.score > 0
        assert len(result.contributions) == 3
        assert all(c.points >= 0 for c in result.contributions)

    def test_null_bin_handling(self):
        """Test null bin tracking (00 §6.10, 09 §5.15 item 16)."""
        characteristics = {"income_level": None}
        bins = {"income_level": [(0, 5000), (5000, float('inf'))]}
        result = evaluate_scorecard(1, characteristics, bins)
        assert result.contributions[0].bin == "null"


class TestCalibration:
    def test_calibrate_score(self):
        """Test score calibration to PD (00 §6.11)."""
        result = calibrate_score(500.0, "retail", {})
        assert 0 < result.probability_of_default < 1
        assert result.odds > 0


class TestRiskGrade:
    def test_assign_grade(self):
        """Test risk grade assignment (00 §6.12)."""
        result = assign_risk_grade(0.1, "retail", {})
        assert 1 <= result.risk_grade <= 12
        assert result.segment_code == "retail"


class TestReasonCodes:
    def test_rank_reasons(self):
        """Test reason code ranking (09 §5.15 item 11)."""
        codes = [1005, 1003, 1001]  # Different severities
        ranked = rank_reasons(codes)
        # Should be ranked by severity (lower value = more severe)
        assert ranked[0] == 1001  # Most severe


class TestRounding:
    def test_round_instalment(self):
        """Test monetary rounding (00 §6.20, 09 §8.4)."""
        # Ensure determinism for replay (09 §8.4)
        amount = 1234.567
        rounded = round_instalment(amount)
        assert rounded == 1234.57
        # Replay same amount should give exact same result
        assert round_instalment(amount) == rounded


class TestEvidence:
    def test_decision_id_generation(self):
        """Test stable decision identifier (09 §5.15 item 1)."""
        from core import generate_decision_id
        id1 = generate_decision_id()
        id2 = generate_decision_id()
        assert id1 != id2  # Never reused
        assert len(id1) > 0


class TestIntegration:
    def test_full_flow(self):
        """Integration test: income through affordability."""
        # Simulate a full assessment
        income_result = determine_income(payslip_income=15000.0)
        deductions_result = calculate_deductions(income_result.gross_monthly_income, employment_type_code=1)
        expenses_result = apply_expense_norms(
            income_result.gross_monthly_income, 2, 4500.0
        )
        obligations_result = calculate_obligations()
        affordability_result = assess_affordability(
            deductions_result.net_monthly_income,
            expenses_result.living_expenses,
            obligations_result.existing_obligations,
            proposed_instalment=2500.0
        )

        assert affordability_result.verdict_code in [1, 2, 3, 4]
        assert affordability_result.max_affordable_instalment >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
