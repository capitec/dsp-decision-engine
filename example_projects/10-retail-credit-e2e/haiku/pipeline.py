"""Retail credit end-to-end pipeline (entry point 1).

§5.23: Loop L1 re-runs affordability when it fails and consolidation is eligible.
All 18 phases and 8 entry points declared as stubs.
Entry point 1 fully implemented for product 10 only.
"""
from __future__ import annotations
from decider import flow, step, param, missing_as, loop, branch
from datetime import date
from typing import Dict, Optional
from flow_state import DecisionState
from phases import (
    p01_request_validation_and_routing,
    p02_client_resolution,
    p03_consent_and_eligibility,
    p04_data_acquisition,
    p05_fraud_assessment,
    p06_feature_derivation,
    p07_scoring,
    p08_calibration_and_grading,
    p09_policy_gates_and_caps,
    p10_affordability_assessment,
    p11_product_routing,
    p12_pricing,
    p13_the_solve,
    p14_consolidation_search,
    p15_limit_assignment,
    p16_offer_assembly,
    p17_final_validation,
    p18_decision_record_emission,
)


@step
def ep1_entry_point_1_new_credit(
    identity_number: Optional[str] = None,
    product_code: int = param(10, ge=10, le=40),
    amount: float = param(50000, ge=2000, le=500000),
    term_months: int = param(36, ge=6, le=84),
    channel_code: int = param(1, ge=1, le=10),
    gross_monthly_income: float = param(30000, ge=1000, le=200000),
    dependants: int = param(0, ge=0, le=10),
    employment_type_code: int = param(1, ge=1, le=6),
    bureau_accounts: int = param(3, ge=0, le=20),
    internal_tenure_months: int = param(24, ge=0, le=360),
    has_bureau_consent: bool = param(True),
    has_data_sharing_consent: bool = param(True),
    consolidation_eligible: bool = param(False),
    decision_date_override: Optional[date] = None
) -> dict:
    """Entry point 1: New credit application.

    §4.1: New credit application. Entry point code 1. 16 of 18 phases,
    P14 conditional. Product 10 only in this implementation.

    Runs: P01-P13, P16-P18.
    Conditional: P14 when affordability fails and consolidation_eligible.
    Loop L1 (§5.23): If P10 fails, run P14 consolidation, then re-run P10-P13.
    """
    # P01: Request validation and routing
    p01_result = p01_request_validation_and_routing(
        entry_point_code=1,
        product_code=product_code,
        amount=amount,
        term_months=term_months,
        channel_code=channel_code,
        consolidation_eligible=consolidation_eligible,
        decision_date_override=decision_date_override
    )

    if not p01_result["validation_pass"]:
        return {
            "outcome": "INVALID_REQUEST",
            "reason": p01_result["rejection_reason"]
        }

    state: DecisionState = p01_result["state"]

    # P02: Client and identity resolution
    p02_result = p02_client_resolution(
        state,
        identity_number=identity_number
    )
    state = p02_result["state"]

    # P03: Consent and hard eligibility
    p03_result = p03_consent_and_eligibility(
        state,
        has_bureau_consent=has_bureau_consent,
        has_data_sharing_consent=has_data_sharing_consent
    )
    state = p03_result["state"]

    if not state.hard_eligibility_pass:
        # Early decline
        state.outcome_code = 2
        state.reason_codes = ["INELIGIBLE"]
        final = p18_decision_record_emission(state)
        return final

    # P04: Data acquisition orchestration
    p04_result = p04_data_acquisition(state)
    state = p04_result["state"]

    # P05: Fraud and financial crime
    p05_result = p05_fraud_assessment(state)
    state = p05_result["state"]

    if state.fraud_verdict_code != 1:
        # Fraud decline
        state.outcome_code = 3
        state.reason_codes = state.fraud_reason_codes
        final = p18_decision_record_emission(state)
        return final

    # P06: Feature derivation
    p06_result = p06_feature_derivation(
        state,
        gross_monthly_income=gross_monthly_income,
        dependants=dependants,
        employment_type_code=employment_type_code,
        bureau_accounts=bureau_accounts,
        internal_tenure_months=internal_tenure_months
    )
    state = p06_result["state"]

    # P07: Scoring
    p07_result = p07_scoring(state)
    state = p07_result["state"]

    # P08: Calibration and grading
    p08_result = p08_calibration_and_grading(state)
    state = p08_result["state"]

    # P09: Policy gates and cap waterfall (first pass, before P10)
    p09_result = p09_policy_gates_and_caps(state)
    state = p09_result["state"]

    # ========================================================================
    # Loop L1 (§5.23): Affordability re-run loop
    # ========================================================================
    # First pass P10 with conservative buffer
    max_loop_passes = 4

    for loop_pass in range(max_loop_passes):
        state.loop_pass_count = loop_pass

        # P10: Affordability (first pass or after consolidation)
        p10_result = p10_affordability_assessment(state)
        state = p10_result["state"]

        if not p10_result["should_trigger_consolidation_loop"]:
            # Affordability passed or can't consolidate; exit loop
            break

        # P14: Consolidation search (only if affordability failed and eligible)
        p14_result = p14_consolidation_search(state)
        state = p14_result["state"]

        # Loop continues: P10 runs again with updated hypothetical obligations
        # (Real implementation would update existing_obligations from chosen scenario)

    # ========================================================================
    # End loop L1
    # ========================================================================

    # P11: Product routing
    p11_result = p11_product_routing(state, request_product=product_code)
    state = p11_result["state"]

    # P12: Pricing
    p12_result = p12_pricing(
        state,
        amount=amount,
        term_months=term_months
    )
    state = p12_result["state"]

    # P13: The solve
    p13_result = p13_the_solve(
        state,
        requested_amount=amount,
        requested_term=term_months
    )
    state = p13_result["state"]

    # P15: Limit assignment (skip for EP1, product 10 is term loan)
    # No call to p15_limit_assignment

    # P16: Offer assembly
    p16_result = p16_offer_assembly(state)
    state = p16_result["state"]

    # P17: Final validation
    p17_result = p17_final_validation(state)
    state = p17_result["state"]

    # P18: Decision record emission
    p18_result = p18_decision_record_emission(state)

    return p18_result


# ============================================================================
# Entry point routing (all 8 stubs for phase set determination)
# ============================================================================

@step
def ep2_client_initiated_limit_change() -> dict:
    """Entry point 2: Client-initiated limit change (stub).

    Products 20, 21. Runs 15 of 18 phases. Skip P11, P13, P14.
    """
    return {"status": "stub", "entry_point": 2, "phases_run": 15}


@step
def ep3_limit_programme_evaluation() -> dict:
    """Entry point 3: Limit programme evaluation (stub).

    Batch, monthly. 4.1M accounts. Runs 14 of 18 phases.
    """
    return {"status": "stub", "entry_point": 3, "phases_run": 14}


@step
def ep4_campaign_preapproval() -> dict:
    """Entry point 4: Campaign pre-approval (stub).

    Batch, monthly. 14.2M clients. Runs 16 of 18, skip P02 (pre-resolved), P05 (fraud).
    """
    return {"status": "stub", "entry_point": 4, "phases_run": 16}


@step
def ep5_consolidation_assessment() -> dict:
    """Entry point 5: Consolidation assessment (stub).

    Interactive. 17 of 18 phases. Skip P15 only.
    """
    return {"status": "stub", "entry_point": 5, "phases_run": 17}


@step
def ep6_reprice_retention() -> dict:
    """Entry point 6: Re-price / retention offer (stub).

    Event-driven. 12 of 18 phases (11 + conditional P10).
    """
    return {"status": "stub", "entry_point": 6, "phases_run": 12}


@step
def ep7_quotation_only() -> dict:
    """Entry point 7: Quotation only, no decision (stub).

    7 of 18 phases: P01, P03 (reduced), P09 (reduced), P11, P12, P17 (reduced), P18.
    Sub-50 ms p99, no external calls.
    """
    return {"status": "stub", "entry_point": 7, "phases_run": 7}


@step
def ep8_what_if_simulation() -> dict:
    """Entry point 8: What-if simulation (stub).

    On demand. Phase set of decision being intervened on.
    Non-confusable identifier required.
    """
    return {"status": "stub", "entry_point": 8, "phases_run": "variable"}


def build(entry_point_code: int = param(1, ge=1, le=8)) -> dict:
    """Build the retail credit end-to-end pipeline.

    §5.20: Eight entry points. This implementation focuses on entry point 1
    (new credit application), with other entry points stubbed for routing.

    §5.21: Shared intermediates registry implemented in DecisionState.
    §5.21.1: Two simultaneous versions of existing_obligations for consolidation scenarios.
    §5.22: Ordering constraints enforced in phase sequence.
    §5.23: Loop L1 in entry point 1 (affordability re-run on consolidation).
    """
    # Route to appropriate entry point
    if entry_point_code == 1:
        return flow(
            ep1_entry_point_1_new_credit,
            name="ep1-new-credit-application"
        )
    elif entry_point_code == 2:
        return flow(ep2_client_initiated_limit_change, name="ep2-limit-change")
    elif entry_point_code == 3:
        return flow(ep3_limit_programme_evaluation, name="ep3-limit-programme")
    elif entry_point_code == 4:
        return flow(ep4_campaign_preapproval, name="ep4-campaign-preapproval")
    elif entry_point_code == 5:
        return flow(ep5_consolidation_assessment, name="ep5-consolidation")
    elif entry_point_code == 6:
        return flow(ep6_reprice_retention, name="ep6-reprice-retention")
    elif entry_point_code == 7:
        return flow(ep7_quotation_only, name="ep7-quotation")
    elif entry_point_code == 8:
        return flow(ep8_what_if_simulation, name="ep8-what-if")
    else:
        # Default to EP1
        return flow(
            ep1_entry_point_1_new_credit,
            name="ep1-new-credit-application"
        )
