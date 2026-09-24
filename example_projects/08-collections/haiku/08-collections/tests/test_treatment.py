"""Tests for collections treatment assignment (project 08)."""
import pytest
from datetime import date
from treatment_assignment.model import TreatmentRequest
from treatment_assignment.assessment import (
    compute_arrears_bucket,
    compute_balance_band,
    compute_collections_score,
    evaluate_suspensions,
    assess_treatment,
)


class TestArreaersBucket:
    """Test arrears bucket assignment."""

    def test_early_bucket(self):
        """Days 1-14 maps to bucket 1."""
        assert compute_arrears_bucket(7, [(0, 14), (15, 29), (30, 59), (60, 89), (90, 119), (120, 179), (180, 364), (365, 10000)]) == 1
        assert compute_arrears_bucket(14, [(0, 14), (15, 29), (30, 59), (60, 89), (90, 119), (120, 179), (180, 364), (365, 10000)]) == 1

    def test_mid_bucket(self):
        """Days 30-59 maps to bucket 3."""
        assert compute_arrears_bucket(45, [(0, 14), (15, 29), (30, 59), (60, 89), (90, 119), (120, 179), (180, 364), (365, 10000)]) == 3

    def test_late_bucket(self):
        """Days 365+ maps to bucket 8."""
        assert compute_arrears_bucket(400, [(0, 14), (15, 29), (30, 59), (60, 89), (90, 119), (120, 179), (180, 364), (365, 10000)]) == 8


class TestBalanceBand:
    """Test balance band assignment."""

    def test_low_balance(self):
        """Balance < 2500 maps to band 1."""
        assert compute_balance_band(1500, [2500, 10000, 25000, 50000, 100000, 250000]) == 1

    def test_mid_balance(self):
        """Balance 25000-50000 maps to band 4."""
        assert compute_balance_band(35000, [2500, 10000, 25000, 50000, 100000, 250000]) == 4

    def test_high_balance(self):
        """Balance >= 250000 maps to band 7."""
        assert compute_balance_band(300000, [2500, 10000, 25000, 50000, 100000, 250000]) == 7


class TestCollectionsScore:
    """Test collections score computation."""

    def test_score_improves_with_payments(self):
        """Score should decrease (improve) with recent payments."""
        request1 = TreatmentRequest(
            account_id=1, client_id=1, decision_date=date(2024, 9, 24),
            days_past_due=30, arrears_amount=5000, balance=25000,
            product_family_code=1, contractual_instalment=1000,
            times_cured_12m=0, times_cured_24m=0,
            right_party_contact_rate=0.2, promise_kept_rate=0.0,
            broken_promises_90d=0, age_months=36, other_accounts_in_arrears=0,
            sms_attempts_90d=3, call_attempts_90d=1, email_attempts_90d=0,
            successful_contacts_90d=0,
            last_payment_date=None, payments_30d=0, payments_60d=0, payments_90d=0
        )

        request2 = TreatmentRequest(
            account_id=2, client_id=2, decision_date=date(2024, 9, 24),
            days_past_due=30, arrears_amount=5000, balance=25000,
            product_family_code=1, contractual_instalment=1000,
            times_cured_12m=0, times_cured_24m=0,
            right_party_contact_rate=0.2, promise_kept_rate=0.0,
            broken_promises_90d=0, age_months=36, other_accounts_in_arrears=0,
            sms_attempts_90d=3, call_attempts_90d=1, email_attempts_90d=0,
            successful_contacts_90d=0,
            last_payment_date=date(2024, 9, 23), payments_30d=1, payments_60d=1, payments_90d=1
        )

        score1, _ = compute_collections_score(request1)
        score2, _ = compute_collections_score(request2)
        assert score2 < score1, "Score should improve with payments"


class TestSuspensions:
    """Test suspension evaluation."""

    def test_no_suspensions(self):
        """Account with no flags should have no suspensions."""
        request = TreatmentRequest(
            account_id=1, client_id=1, decision_date=date(2024, 9, 24),
            days_past_due=30, arrears_amount=5000, balance=25000,
            product_family_code=1, contractual_instalment=1000,
            times_cured_12m=0, times_cured_24m=0,
            right_party_contact_rate=0.2, promise_kept_rate=0.0,
            broken_promises_90d=0, age_months=36, other_accounts_in_arrears=0,
            sms_attempts_90d=0, call_attempts_90d=0, email_attempts_90d=0,
            successful_contacts_90d=0,
            debt_review_status_code=None,
            deceased=False,
            dispute_raised=False
        )
        suspensions, blocks_all = evaluate_suspensions(request)
        assert len(suspensions) == 0
        assert not blocks_all

    def test_debt_review_suspension(self):
        """Debt review status should trigger suspension."""
        request = TreatmentRequest(
            account_id=1, client_id=1, decision_date=date(2024, 9, 24),
            days_past_due=30, arrears_amount=5000, balance=25000,
            product_family_code=1, contractual_instalment=1000,
            times_cured_12m=0, times_cured_24m=0,
            right_party_contact_rate=0.2, promise_kept_rate=0.0,
            broken_promises_90d=0, age_months=36, other_accounts_in_arrears=0,
            sms_attempts_90d=0, call_attempts_90d=0, email_attempts_90d=0,
            successful_contacts_90d=0,
            debt_review_status_code=101,  # Application received
            deceased=False,
            dispute_raised=False
        )
        suspensions, blocks_all = evaluate_suspensions(request)
        assert 101 in suspensions
        assert blocks_all

    def test_deceased_suspension(self):
        """Deceased flag should trigger suspension."""
        request = TreatmentRequest(
            account_id=1, client_id=1, decision_date=date(2024, 9, 24),
            days_past_due=30, arrears_amount=5000, balance=25000,
            product_family_code=1, contractual_instalment=1000,
            times_cured_12m=0, times_cured_24m=0,
            right_party_contact_rate=0.2, promise_kept_rate=0.0,
            broken_promises_90d=0, age_months=36, other_accounts_in_arrears=0,
            sms_attempts_90d=0, call_attempts_90d=0, email_attempts_90d=0,
            successful_contacts_90d=0,
            debt_review_status_code=None,
            deceased=True,
            dispute_raised=False
        )
        suspensions, blocks_all = evaluate_suspensions(request)
        assert 107 in suspensions
        assert blocks_all


class TestFullAssessment:
    """End-to-end assessment tests."""

    def test_healthy_account_gets_sms(self):
        """Healthy early-bucket account should get SMS."""
        request = TreatmentRequest(
            account_id=1, client_id=1, decision_date=date(2024, 9, 24),
            days_past_due=10, arrears_amount=1000, balance=25000,
            product_family_code=1, contractual_instalment=1000,
            times_cured_12m=1, times_cured_24m=2,
            right_party_contact_rate=0.5, promise_kept_rate=0.8,
            broken_promises_90d=0, age_months=48, other_accounts_in_arrears=0,
            sms_attempts_90d=2, call_attempts_90d=0, email_attempts_90d=0,
            successful_contacts_90d=2,
            last_payment_date=date(2024, 9, 20),
            payments_30d=1, payments_60d=1, payments_90d=2
        )
        result = assess_treatment(request)
        assert result.treatment_code in (1, 2, 3, 4), "Early bucket should get automated treatment"
        assert result.non_selection_reason_code is None or result.non_selection_reason_code == 200

    def test_suspended_account_blocked(self):
        """Suspended account should not receive treatment."""
        request = TreatmentRequest(
            account_id=2, client_id=2, decision_date=date(2024, 9, 24),
            days_past_due=90, arrears_amount=10000, balance=50000,
            product_family_code=1, contractual_instalment=1000,
            times_cured_12m=0, times_cured_24m=0,
            right_party_contact_rate=0.2, promise_kept_rate=0.2,
            broken_promises_90d=2, age_months=36, other_accounts_in_arrears=1,
            sms_attempts_90d=5, call_attempts_90d=3, email_attempts_90d=1,
            successful_contacts_90d=0,
            debt_review_status_code=101  # Debt review
        )
        result = assess_treatment(request)
        assert 101 in result.active_suspensions
        assert result.treatment_code == 0, "Suspended account should get no treatment"

    def test_evidence_recorded(self):
        """Assessment should record full evidence (§5.15)."""
        request = TreatmentRequest(
            account_id=3, client_id=3, decision_date=date(2024, 9, 24),
            days_past_due=45, arrears_amount=5000, balance=30000,
            product_family_code=1, contractual_instalment=1200,
            times_cured_12m=1, times_cured_24m=1,
            right_party_contact_rate=0.4, promise_kept_rate=0.5,
            broken_promises_90d=1, age_months=40, other_accounts_in_arrears=0,
            sms_attempts_90d=4, call_attempts_90d=2, email_attempts_90d=0,
            successful_contacts_90d=1
        )
        result = assess_treatment(request)

        # Evidence checklist (§5.15 items)
        assert result.decision_id is not None  # Item 1: stable identifier
        assert result.matrix_cell_id is not None  # Item 5: table cell attribution
        assert result.decision_date == request.decision_date  # Item 4: no reliance on "today"
        assert result.matrix_version >= 0  # Item 5: reference data versions
        assert result.collections_score_unadjusted is not None  # Item 7: unadjusted values
        assert result.episode_id > 0 or request.episode_id == 0  # Item 8: state snapshotted
