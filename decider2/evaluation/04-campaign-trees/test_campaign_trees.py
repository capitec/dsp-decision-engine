"""Tests for the campaign targeting trees implementation.

Tests the simplified campaign tree that exercises:
- Multi-level decision trees
- Sequential decision points
- Multiple outcomes
- Equivalence across execution modes
"""
import sys
from pathlib import Path

# Add the decider2 shim package to path
# File location: /home/sholto/.../decider2/evaluation/04-campaign-trees/test_campaign_trees.py
# Want to add: /home/sholto/.../decider2 (the shim dir with decider2/__init__.py and decider2/)
_decider2_shim = Path(__file__).parent.parent.parent.resolve()
if str(_decider2_shim) not in sys.path:
    sys.path.insert(0, str(_decider2_shim))

import polars as pl
import pytest

from decider2.testing import assert_equivalent

from pipeline import pipeline


# Test data covering various paths through the tree
FRAME = pl.DataFrame({
    "has_active_flex_loan": [True, True, False, True, True, True],
    "months_on_book_flex": [12, 12, 12, 2, 12, 12],
    "settlement_ratio": [0.5, 0.8, 0.5, 0.5, 0.5, 0.5],
    "payments_missed_12m": [0, 1, 0, 0, 0, 0],
    "behaviour_score": [700.0, 700.0, 700.0, 700.0, 500.0, 700.0],
    "discretionary_income": [3500.0, 3500.0, 3500.0, 3500.0, 3500.0, 1500.0],
    "estimated_instalment_to_income": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
})


class TestBasicFunctionality:
    """Test basic tree evaluation."""

    def test_single_record_targeting(self):
        """Test that a qualifying client is targeted."""
        result = pipeline.score({
            "has_active_flex_loan": True,
            "months_on_book_flex": 12,
            "settlement_ratio": 0.5,
            "payments_missed_12m": 0,
            "behaviour_score": 700.0,
            "discretionary_income": 3500.0,
            "estimated_instalment_to_income": 0.25,
        })
        assert result["targeting_decision"] == 1

    def test_facility_check_fail(self):
        """Test that client without facility is rejected early."""
        result = pipeline.score({
            "has_active_flex_loan": False,
            "months_on_book_flex": 12,
            "settlement_ratio": 0.5,
            "payments_missed_12m": 0,
            "behaviour_score": 700.0,
            "discretionary_income": 3500.0,
            "estimated_instalment_to_income": 0.25,
        })
        assert result["targeting_decision"] == 0

    def test_payment_history_fail(self):
        """Test that client with poor payment history is rejected."""
        result = pipeline.score({
            "has_active_flex_loan": True,
            "months_on_book_flex": 12,
            "settlement_ratio": 0.8,  # Too high
            "payments_missed_12m": 0,
            "behaviour_score": 700.0,
            "discretionary_income": 3500.0,
            "estimated_instalment_to_income": 0.25,
        })
        assert result["targeting_decision"] == 0

    def test_credit_score_fail(self):
        """Test that client with low credit score is rejected."""
        result = pipeline.score({
            "has_active_flex_loan": True,
            "months_on_book_flex": 12,
            "settlement_ratio": 0.5,
            "payments_missed_12m": 0,
            "behaviour_score": 500.0,  # Below 600
            "discretionary_income": 3500.0,
            "estimated_instalment_to_income": 0.25,
        })
        assert result["targeting_decision"] == 0

    def test_affordability_fail(self):
        """Test that client with insufficient income is rejected."""
        result = pipeline.score({
            "has_active_flex_loan": True,
            "months_on_book_flex": 12,
            "settlement_ratio": 0.5,
            "payments_missed_12m": 0,
            "behaviour_score": 700.0,
            "discretionary_income": 1500.0,  # Below 2500
            "estimated_instalment_to_income": 0.25,
        })
        assert result["targeting_decision"] == 0


class TestBatchMode:
    """Test batch processing."""

    def test_batch_processing_works(self):
        """Test that the pipeline processes batches correctly."""
        result = pipeline.apply(FRAME)
        assert len(result) == 6
        assert "targeting_decision" in result.columns

    def test_batch_mixed_decisions(self):
        """Test that batch produces mixed targeting decisions."""
        result = pipeline.apply(FRAME)
        decisions = result["targeting_decision"].to_list()
        # Should have a mix of 0s and 1s due to different paths
        assert 0 in decisions
        assert any(d == 1 for d in decisions)


class TestEquivalenceMode:
    """Test the three execution modes agree - core test from doc 02 section 3.1."""

    def test_modes_agree_exactly(self):
        """Test that interpreted, stepped, and fused modes agree exactly.

        This is the core equivalence requirement from doc 02 section 3.1
        and doc 05 section 9.1. The assert_equivalent function runs all three
        modes and verifies they produce identical output.
        """
        assert_equivalent(pipeline, FRAME)
