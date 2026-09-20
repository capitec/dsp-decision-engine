"""Tests for fraud interdiction pipeline."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import polars as pl
from pipelines.fraud_interdiction import (
    pipeline,
    SharedParams,
    cf_0101_velocity_spike,
    cf_0102_amount_spike,
    ms_0208_first_payment_new_beneficiary,
)


class TestCardFraudRules:
    """Test individual card fraud rules."""

    def test_cf_0101_velocity_spike_no_history(self):
        """Transaction count exceeding threshold with no history."""
        assert cf_0101_velocity_spike(
            transaction_count_1h=15,
            historic_max_1h=0,
        )
        assert not cf_0101_velocity_spike(
            transaction_count_1h=5,
            historic_max_1h=0,
        )

    def test_cf_0101_velocity_spike_with_history(self):
        """Transaction count exceeding multiplied historic max."""
        assert cf_0101_velocity_spike(
            transaction_count_1h=51,
            historic_max_1h=10,
            threshold_multiplier=5.0,
        )
        assert not cf_0101_velocity_spike(
            transaction_count_1h=49,
            historic_max_1h=10,
            threshold_multiplier=5.0,
        )

    def test_cf_0102_amount_spike(self):
        """Single transaction amount above threshold."""
        assert cf_0102_amount_spike(
            transaction_amount=30001.0,
            historic_max_amount=10000.0,
            spike_factor=3.0,
        )
        assert not cf_0102_amount_spike(
            transaction_amount=29999.0,
            historic_max_amount=10000.0,
            spike_factor=3.0,
        )


class TestMuleScamRules:
    """Test mule and scam detection rules."""

    def test_ms_0208_first_payment_new_beneficiary(self):
        """First payment to newly added beneficiary with risk factors."""
        assert ms_0208_first_payment_new_beneficiary(
            beneficiary_age_hours=1,
            transaction_amount=10000.0,
            device_change_hours=12,
        )
        assert not ms_0208_first_payment_new_beneficiary(
            beneficiary_age_hours=5,
            transaction_amount=10000.0,
            device_change_hours=12,
        )


class TestPipelineEndToEnd:
    """End-to-end pipeline tests."""

    @pytest.fixture
    def sample_record(self):
        """Sample fraud event record."""
        return {
            # Card fraud features
            "transaction_count_1h": 25,
            "historic_max_1h": 5,
            "transaction_amount": 15000.0,
            "historic_max_amount": 5000.0,
            "merchant_risk_band": 5,
            "client_prior_merchants": 0,
            # Mule/scam features
            "beneficiary_age_hours": 1,
            "device_change_hours": 12,
            "beneficiary_count_24h": 3,
            "unique_beneficiaries_7d": 8,
            # Account takeover features
            "transaction_count_since_device_change": 25,
            "hours_since_device_change": 6,
            # Model score
            "model_score": 800.0,
        }

    def test_pipeline_detects_violations(self, sample_record):
        """Pipeline correctly identifies rule violations."""
        result = pipeline.score(sample_record)

        assert result["fired_rule_count"] > 0
        assert "max_severity" in result

    def test_pipeline_clean_record(self):
        """Pipeline allows clean record through."""
        clean_record = {
            "transaction_count_1h": 2,
            "historic_max_1h": 5,
            "transaction_amount": 500.0,
            "historic_max_amount": 5000.0,
            "merchant_risk_band": 2,
            "client_prior_merchants": 5,
            "beneficiary_age_hours": 100,
            "device_change_hours": 999,
            "beneficiary_count_24h": 0,
            "unique_beneficiaries_7d": 2,
            "transaction_count_since_device_change": 1,
            "hours_since_device_change": 48,
            "model_score": 900.0,
        }
        result = pipeline.score(clean_record)

        assert result["fired_rule_count"] == 0
        assert result["max_severity"] == 0

    def test_pipeline_batch_processing(self, sample_record):
        """Pipeline handles batch processing correctly."""
        df = pl.DataFrame(
            [sample_record, {**sample_record, "model_score": 650.0}]
        )
        result = pipeline.apply(df)

        assert len(result) == 2
        assert "fired_rule_count" in result.columns
        assert "max_severity" in result.columns


class TestEquivalenceLadder:
    """Test execution modes equivalence."""

    @pytest.fixture
    def test_records(self):
        """Create test records for equivalence testing."""
        return [
            {
                "transaction_count_1h": 20,
                "historic_max_1h": 4,
                "transaction_amount": 12000.0,
                "historic_max_amount": 4000.0,
                "merchant_risk_band": 4,
                "client_prior_merchants": 1,
                "beneficiary_age_hours": 1,
                "device_change_hours": 12,
                "beneficiary_count_24h": 2,
                "unique_beneficiaries_7d": 5,
                "transaction_count_since_device_change": 30,
                "hours_since_device_change": 8,
                "model_score": 780.0,
            },
            {
                "transaction_count_1h": 3,
                "historic_max_1h": 5,
                "transaction_amount": 1000.0,
                "historic_max_amount": 8000.0,
                "merchant_risk_band": 2,
                "client_prior_merchants": 10,
                "beneficiary_age_hours": 200,
                "device_change_hours": 500,
                "beneficiary_count_24h": 1,
                "unique_beneficiaries_7d": 3,
                "transaction_count_since_device_change": 5,
                "hours_since_device_change": 72,
                "model_score": 950.0,
            },
        ]

    def test_execution_modes_agree(self, test_records):
        """All execution modes produce identical results."""
        from decider2.testing import assert_equivalent

        df = pl.DataFrame(test_records)

        assert_equivalent(
            pipeline,
            df,
            origin="test",
        )
