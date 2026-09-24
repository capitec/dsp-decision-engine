"""Tests for EP-3 annual review and covenant tests."""
import pytest
import sys
from pathlib import Path
from datetime import date

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from business_credit_e2e.pipeline import ep3_annual_review, covenant_test_dscr
from business_credit_e2e.assessment import (
    assess_annual_review,
    covenant_dscr_test,
    PriorDecision,
)


def test_ep3_annual_review_no_prior():
    """Test annual review without prior decision (first review)."""
    result = ep3_annual_review(
        facility_id=200001,
        client_id=100001,
    )

    assert result["facility_id"] == 200001
    assert result["grade_migration"] is None
    assert result["business_grade"] == 6


def test_ep3_annual_review_with_grade_migration():
    """Test annual review with grade improvement."""
    result = ep3_annual_review(
        facility_id=200002,
        client_id=100002,
        prior_grade=7,
        prior_decision_id=1001,
    )

    assert result["facility_id"] == 200002
    assert result["prior_grade"] == 7
    assert result["prior_decision_id"] == 1001
    # Grade migration should be: new grade (6) - prior grade (7) = -1
    assert result["grade_migration"] == -1


def test_covenant_dscr_pass():
    """Test DSCR covenant pass scenario."""
    result = covenant_test_dscr(
        facility_id=300001,
        ebitda=150000,
        debt_service_annual=100000,
        dscr_threshold=1.25,
    )

    assert result["facility_id"] == 300001
    assert result["covenant_type"] == "DSCR"
    assert result["measured_value"] == pytest.approx(1.5)
    assert result["breach_class_code"] == 0  # No breach
    assert result["headroom_pct"] == pytest.approx(20.0)


def test_covenant_dscr_material_breach():
    """Test DSCR covenant material breach."""
    result = covenant_test_dscr(
        facility_id=300002,
        ebitda=100000,
        debt_service_annual=100000,
        dscr_threshold=1.25,
    )

    assert result["facility_id"] == 300002
    assert result["measured_value"] == pytest.approx(1.0)
    assert result["breach_class_code"] == 2  # Material


def test_covenant_dscr_severe_breach():
    """Test DSCR covenant severe breach."""
    result = covenant_test_dscr(
        facility_id=300003,
        ebitda=80000,
        debt_service_annual=100000,
        dscr_threshold=1.25,
    )

    assert result["facility_id"] == 300003
    assert result["measured_value"] == pytest.approx(0.8)
    assert result["breach_class_code"] == 4  # Severe


def test_covenant_dscr_definition_version_pinned():
    """Test that DSCR definition version is pinned."""
    result = covenant_test_dscr(
        facility_id=300004,
    )

    # Definition version should remain unchanged even if policy updates
    # Current implementation uses hardcoded version
    assert result["definition_version"] == "2026-01"


def test_assess_annual_review_direct():
    """Test assess_annual_review function directly."""
    prior = PriorDecision(
        decision_id=1001,
        decision_date=date(2025, 1, 15),
        business_grade=7,
        master_scale_version="v1.0",
        overlay_stack_unadjusted_grade=7,
    )

    result = assess_annual_review(
        facility_id=200003,
        client_id=100003,
        decision_date=date(2026, 1, 15),
        review_date=date(2026, 1, 15),
        entities_list=[],
        adverse_events_list=[],
        prior_decision=prior,
    )

    assert result.prior_decision_id == 1001
    assert result.grade_migration == -1  # 6 - 7


def test_assess_annual_review_scale_change():
    """Test that scale change is identified."""
    prior = PriorDecision(
        decision_id=1002,
        decision_date=date(2025, 1, 15),
        business_grade=6,
        master_scale_version="v0.9",  # Different version
        overlay_stack_unadjusted_grade=6,
    )

    result = assess_annual_review(
        facility_id=200004,
        client_id=100004,
        decision_date=date(2026, 1, 15),
        review_date=date(2026, 1, 15),
        entities_list=[],
        adverse_events_list=[],
        prior_decision=prior,
    )

    assert "MASTER_SCALE_CHANGE" in result.grade_change_reason_codes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
