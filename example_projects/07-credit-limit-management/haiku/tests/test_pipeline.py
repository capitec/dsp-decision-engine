"""Tests for credit limit management pipeline."""
import pytest
from datetime import date
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline import build, limit_decision


class TestLimitDecision:
    """Test the consolidated limit_decision step."""

    def test_basic_decision(self):
        """Test basic limit decision with clean account."""
        result = limit_decision(
            account_id=12345,
            product_code=20,
            client_id=98765,
            current_limit=15000.0,
            current_balance=7500.0,
            months_on_book=24,
            revolving_utilisation_6m=0.50,
            arrears_worst_3m=0,
            arrears_worst_24m=0,
            cycles_ever_past_due=0,
            over_limit_count_12m=0,
            cash_withdrawal_ratio_12m=0.20,
            spend_percentile_90_6m=800.0,
            has_auto_increase_consent=True,
            treatment_state_code=0,
            gross_monthly_income=10000.0,
            existing_obligations=2000.0,
            decision_date=date(2026, 9, 24)
        )

        assert result['account_id'] == 12345
        assert result['is_excluded'] is False
        assert result['behaviour_grade'] in range(1, 13)
        assert result['proposed_limit'] >= 15000.0  # Should not decrease
        assert result['affordability_verdict_code'] in [1, 3]  # Pass or fail

    def test_exclusion_no_consent(self):
        """Test account excluded for no consent."""
        result = limit_decision(
            has_auto_increase_consent=False,
            months_on_book=24
        )

        assert result['is_excluded'] is True
        assert 12 in result['exclusion_codes']

    def test_exclusion_too_young(self):
        """Test account excluded for being too young."""
        result = limit_decision(
            months_on_book=3  # < 6 months
        )

        assert result['is_excluded'] is True
        assert 14 in result['exclusion_codes']


class TestPipelineIntegration:
    """Integration tests for full pipeline."""

    def test_build_pipeline(self):
        """Test pipeline builds without error."""
        pipeline = build()
        assert pipeline is not None
        assert hasattr(pipeline, 'name')
        assert pipeline.name == 'credit_limit_management'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
