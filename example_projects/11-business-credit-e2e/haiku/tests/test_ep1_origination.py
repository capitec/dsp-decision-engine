"""Tests for EP-1 origination (business credit application)."""
import pytest
import sys
from pathlib import Path
from datetime import date

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from business_credit_e2e.pipeline import ep1_origination
from business_credit_e2e.assessment import assess_origination


def test_ep1_basic_origination():
    """Test EP-1 with minimal inputs."""
    result = ep1_origination(
        application_id=100001,
        product_code=50,
        amount=500000,
        term_months=36,
        entities_list=[],
        adverse_events_list=[],
    )

    assert result["application_id"] == 100001
    assert result["product_code"] == 50
    assert result["business_grade"] == 6
    assert result["business_verdict_code"] == 1


def test_ep1_product_51_revolving():
    """Test product 51 (revolving)."""
    result = ep1_origination(
        application_id=100002,
        product_code=51,
        amount=1000000,
        term_months=24,
    )

    assert result["product_code"] == 51
    assert result["amount"] == 1000000


def test_assess_origination_function():
    """Test assess_origination directly."""
    result = assess_origination(
        application_id=100003,
        decision_date=date(2026, 1, 15),
        product_code=50,
        entities_list=[],
        adverse_events_list=[],
    )

    assert result.application_id == 100003
    assert result.decision_date == date(2026, 1, 15)
    assert result.business_grade in range(1, 13)
