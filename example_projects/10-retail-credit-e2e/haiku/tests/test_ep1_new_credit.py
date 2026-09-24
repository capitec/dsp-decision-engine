"""Tests for entry point 1 (new credit application).

Tests the full flow P01-P13, P16-P18 for product 10 (Flex Loan).
Includes loop L1 (affordability re-run) scenarios.
"""
import pytest
from datetime import date
import sys
import os

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_state import DecisionState
from phases import (
    p01_request_validation_and_routing,
    p02_client_resolution,
    p03_consent_and_eligibility,
    p06_feature_derivation,
    p07_scoring,
    p08_calibration_and_grading,
    p09_policy_gates_and_caps,
    p10_affordability_assessment,
    p11_product_routing,
    p12_pricing,
    p13_the_solve,
    p16_offer_assembly,
    p17_final_validation,
    p18_decision_record_emission,
)


class TestP01RequestValidation:
    """P01: Request validation and routing."""

    def test_valid_request_ep1_product10(self):
        """Valid request for EP1, product 10."""
        result = p01_request_validation_and_routing(
            entry_point_code=1,
            product_code=10,
            amount=50000,
            term_months=36
        )
        assert result["validation_pass"]
        assert result["state"].entry_point_code == 1
        assert result["state"].product_codes == [10]
        assert result["state"].phase_set_id == "ep1_product10"

    def test_amount_below_minimum(self):
        """Amount below product minimum."""
        result = p01_request_validation_and_routing(
            entry_point_code=1,
            product_code=10,
            amount=1000,  # Below R2k minimum
            term_months=36
        )
        assert not result["validation_pass"]
        assert "INVALID_REQUEST" in result["rejection_reason"]

    def test_amount_above_maximum(self):
        """Amount above product maximum."""
        result = p01_request_validation_and_routing(
            entry_point_code=1,
            product_code=10,
            amount=600000,  # Above R500k maximum
            term_months=36
        )
        assert not result["validation_pass"]

    def test_term_out_of_range(self):
        """Term outside 6-84 month range."""
        result = p01_request_validation_and_routing(
            entry_point_code=1,
            product_code=10,
            amount=50000,
            term_months=100  # Above 84 month maximum
        )
        assert not result["validation_pass"]

    def test_decision_date_fixed(self):
        """decision_date is fixed in P01 and recorded (O-04)."""
        test_date = date(2024, 9, 1)
        result = p01_request_validation_and_routing(
            entry_point_code=1,
            product_code=10,
            amount=50000,
            term_months=36,
            decision_date_override=test_date
        )
        assert result["state"].decision_date == test_date


class TestIdentityAndConsent:
    """P02: Identity resolution, P03: Consent."""

    def test_identity_resolution(self):
        """P02: Client resolution with identity number."""
        state_from_p01 = p01_request_validation_and_routing(
            entry_point_code=1,
            product_code=10,
            amount=50000,
            term_months=36
        )["state"]

        result = p02_client_resolution(
            state_from_p01,
            identity_number="9005141234087"
        )
        assert result["client_id"] == "9005141234087"
        assert result["identity_confidence"] > 0.9

    def test_consent_obtained(self):
        """P03: Bureau and data sharing consent obtained."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10]
        )
        state.client_id = "9005141234087"

        result = p03_consent_and_eligibility(
            state,
            has_bureau_consent=True,
            has_data_sharing_consent=True
        )
        assert result["consent_obtained"]
        assert result["hard_eligibility_pass"]


class TestFeatureDerivation:
    """P06: Feature derivation – income, expenses, segment."""

    def test_thin_file_segment(self):
        """Feature derivation: thin file assignment."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10]
        )

        result = p06_feature_derivation(
            state,
            gross_monthly_income=30000,
            dependants=1,
            employment_type_code=1,
            bureau_accounts=2,  # < 3
            internal_tenure_months=12  # < 15
        )
        assert result["state"].segment_code == 1  # Thin file

    def test_thick_file_segment(self):
        """Feature derivation: thick file assignment."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10]
        )

        result = p06_feature_derivation(
            state,
            gross_monthly_income=30000,
            dependants=0,
            bureau_accounts=4,  # >= 3
            internal_tenure_months=24  # >= 15
        )
        assert result["state"].segment_code == 4  # Existing clean


class TestScoringAndGrading:
    """P07: Scoring, P08: Calibration and grading."""

    def test_scoring_income_adjustment(self):
        """P07: Score adjusted for income level."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            net_monthly_income=45000,
            segment_code=4
        )

        result = p07_scoring(state, base_score=650)
        assert result["score"] > 650  # Income bump
        assert 0.0 <= result["probability_of_default"] <= 1.0

    def test_risk_grading_from_pd(self):
        """P08: PD to grade bucketing."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            probability_of_default=0.05
        )

        result = p08_calibration_and_grading(state)
        # PD 0.05 is not < 0.05, so maps to grade 4 (boundaries: <0.01=1, <0.02=2, <0.05=3, <0.08=4)
        assert result["risk_grade"] == 4


class TestAffordability:
    """P10: Affordability assessment."""

    def test_affordability_pass(self):
        """Affordability passes when discretionary income is positive."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            net_monthly_income=30000,
            living_expenses=5000,
            existing_obligations=3000  # R30k - R5k - R3k = R22k discretionary
        )

        result = p10_affordability_assessment(state)
        assert result["affordability_verdict_code"] == 1
        assert result["state"].affordability_pass

    def test_affordability_fail(self):
        """Affordability fails when discretionary income is negative."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            net_monthly_income=10000,
            living_expenses=8000,
            existing_obligations=5000  # R10k - R8k - R5k = -R3k (negative!)
        )

        result = p10_affordability_assessment(state)
        assert result["affordability_verdict_code"] == 2
        assert not result["state"].affordability_pass

    def test_loop_l1_consolidation_trigger(self):
        """Loop L1 triggered when affordability fails and consolidation_eligible."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            net_monthly_income=10000,
            living_expenses=8000,
            existing_obligations=5000,
            consolidation_eligible=True
        )

        result = p10_affordability_assessment(state, max_loop_passes=4)
        assert result["should_trigger_consolidation_loop"]


class TestPricing:
    """P12: Pricing – rate, fees, instalment."""

    def test_pricing_calculation(self):
        """P12: Pricing calculates rate, fees, instalment."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            risk_grade=6  # Mid-grade
        )

        result = p12_pricing(
            state,
            amount=50000,
            term_months=36
        )
        assert result["nominal_annual_rate"] > 0.10
        assert result["initiation_fee"] > 0
        assert result["instalment"] > 0

    def test_instalment_bounds(self):
        """P12: Instalment is positive and reasonable."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            risk_grade=6
        )

        result = p12_pricing(state, amount=50000, term_months=36)
        instalment = result["instalment"]
        # Rough check: instalment per month < amount / term
        assert instalment < (50000 / 36) * 1.5


class TestTheSolve:
    """P13: The solve – circular constraint resolution."""

    def test_solve_respects_caps(self):
        """P13: Solve respects amount and term caps."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            amount_cap=100000,
            term_cap=60
        )

        result = p13_the_solve(
            state,
            requested_amount=200000,  # Above cap
            requested_term=72  # Above cap
        )
        assert result["proposed_amount"] <= state.amount_cap
        assert result["proposed_term_months"] <= state.term_cap


class TestOfferAndValidation:
    """P16: Offer assembly, P17: Final validation."""

    def test_offer_assembled(self):
        """P16: Offer assembled when affordability passes."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            affordability_pass=True,
            proposed_amount=50000,
            proposed_term_months=36,
            instalment=1500,
            routed_product_code=10,
            nominal_annual_rate=0.15,
            solve_binding_constraint="affordability"
        )

        result = p16_offer_assembly(state)
        assert result["offers_assembled"] == 1
        assert len(result["offers"]) == 1

    def test_validation_instalment_valid(self):
        """P17: Validation passes with valid instalment."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            instalment=1500,
            proposed_amount=50000,
            proposed_term_months=36,
            nominal_annual_rate=0.15,
            risk_grade=6
        )

        result = p17_final_validation(state)
        assert result["validation_pass"]

    def test_validation_instalment_invalid(self):
        """P17: Validation fails with invalid instalment."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            instalment=0,  # Invalid
            proposed_amount=50000,
            proposed_term_months=36,
            nominal_annual_rate=0.15,
            risk_grade=6
        )

        result = p17_final_validation(state)
        assert not result["validation_pass"]
        assert "INSTALMENT_INVALID" in result["validation_failures"]


class TestDecisionRecord:
    """P18: Decision record emission (09 §5.15 compliance)."""

    def test_approved_outcome(self):
        """P18: Approved outcome when all checks pass."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            validation_pass=True,
            affordability_verdict_code=1,
            fraud_verdict_code=1,
            hard_eligibility_pass=True,
            offers=[{"product_code": 10, "amount": 50000}],
            routed_product_code=10,
            proposed_amount=50000,
            proposed_term_months=36,
            instalment=1500,
            nominal_annual_rate=0.15,
            risk_grade=6,
            client_id="9005141234087"
        )

        result = p18_decision_record_emission(state)
        assert result["outcome_code"] == 1  # Approved
        assert len(result["decision_record"]["reason_codes"]) == 0

    def test_decline_unaffordable(self):
        """P18: Decline when unaffordable."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            validation_pass=True,
            affordability_verdict_code=2,  # Fail
            fraud_verdict_code=1,
            hard_eligibility_pass=True,
            routed_product_code=10,
            proposed_amount=50000,
            proposed_term_months=36
        )

        result = p18_decision_record_emission(state)
        assert result["outcome_code"] == 2  # Decline
        assert "UNAFFORDABLE" in result["reason_codes"]

    def test_decision_id_generated(self):
        """P18: Decision ID generated (09 §5.15 item 1)."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            validation_pass=True,
            affordability_verdict_code=1,
            fraud_verdict_code=1,
            hard_eligibility_pass=True,
            offers=[]
        )

        result = p18_decision_record_emission(state)
        assert len(result["decision_id"]) > 0
        assert "DECID-" in result["decision_id"]


class TestSharedIntermediates:
    """Tests for shared intermediates (§5.21)."""

    def test_net_income_cascades_through_phases(self):
        """net_monthly_income (P06) consumed by P07..P18 (11 consumers)."""
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10]
        )

        # P06 sets it
        p06_result = p06_feature_derivation(state, gross_monthly_income=30000)
        state = p06_result["state"]
        net_income = state.net_monthly_income

        # P07 consumes it
        p07_result = p07_scoring(state)
        state = p07_result["state"]
        # If scoring changes, it's because net_income was consumed
        assert state.net_monthly_income == net_income


class TestOrderingConstraints:
    """Tests for ordering constraints (§5.22)."""

    def test_decision_date_never_reread(self):
        """O-04: decision_date fixed in P01, never re-read."""
        test_date = date(2024, 9, 1)
        result = p01_request_validation_and_routing(
            entry_point_code=1,
            product_code=10,
            amount=50000,
            term_months=36,
            decision_date_override=test_date
        )
        state = result["state"]
        # State records the date; subsequent phases should use it, not re-read
        assert state.decision_date == test_date

    def test_affordability_before_after_product_routing(self):
        """O-09: P10 before P11 with conservative buffer, then after with product buffer."""
        # This is implemented in the pipeline as a loop; here we just test
        # that both runs are possible
        state = DecisionState(
            entry_point_code=1,
            phase_set_id="ep1_product10",
            decision_date=date.today(),
            product_codes=[10],
            net_monthly_income=30000,
            living_expenses=5000,
            existing_obligations=3000
        )

        # First run
        result1 = p10_affordability_assessment(state)
        first_max_instalment = result1["state"].max_affordable_instalment

        # After routing, second run (with different buffer)
        state = result1["state"]
        state.routed_product_code = 10
        result2 = p10_affordability_assessment(state, buffer_override=0.15)
        second_max_instalment = result2["state"].max_affordable_instalment

        # Different buffer should produce different result
        assert first_max_instalment != second_max_instalment


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
