"""
Collections treatment assignment pipeline (project 08).

Assigns daily treatment to delinquent accounts based on account state, suspensions,
collections score, and treatment matrix lookup. Tracks episode position and
capacity constraints.
"""
from __future__ import annotations
from datetime import date
from typing import Optional, List
from decider import flow, param, missing_as

from treatment_assignment.model import TreatmentRequest
from treatment_assignment.assessment import assess_treatment


def assign_treatment(
    # Account identity
    account_id: int = param(1000001),
    client_id: int = param(100001),
    decision_date: date = missing_as(date.today()),

    # Delinquency state
    days_past_due: int = param(30),
    arrears_amount: float = param(5000.0),
    balance: float = param(25000.0),

    # Account characteristics
    product_family_code: int = param(1),
    contractual_instalment: float = param(1000.0),

    # Collections score inputs
    times_cured_12m: int = param(1),
    times_cured_24m: int = param(2),
    right_party_contact_rate: float = param(0.3),
    promise_kept_rate: float = param(0.5),
    broken_promises_90d: int = param(1),
    age_months: int = param(36),
    other_accounts_in_arrears: int = param(1),

    # Contact history
    sms_attempts_90d: int = param(5),
    call_attempts_90d: int = param(2),
    email_attempts_90d: int = param(1),
    successful_contacts_90d: int = param(1),

    # Payment history
    last_payment_date: Optional[date] = None,
    last_payment_amount: float = param(500.0),
    payments_30d: int = param(0),
    payments_60d: int = param(1),
    payments_90d: int = param(2),

    # Arrangement history
    active_arrangement: bool = param(False),
    arrangement_id: Optional[int] = None,
    consecutive_failed_arrangements: int = param(0),

    # Status flags
    debt_review_status_code: Optional[int] = None,
    hardship_arrangement: bool = param(False),
    deceased: bool = param(False),
    external_debt_counselling: bool = param(False),
    dispute_raised: bool = param(False),
    litigation_in_progress: bool = param(False),

    # Episode tracking
    episode_id: int = param(0),
    previous_treatment_code: Optional[int] = None,
    previous_treatment_date: Optional[date] = None,
    channel_attempt_count: int = param(0),
    intensity_ceiling: int = param(1),
) -> dict:
    """
    Assign treatment to a delinquent account.

    This step:
    1. Assembles account state (§5.1)
    2. Evaluates suspensions (§5.2)
    3. Computes collections score and bands (§5.3)
    4. Looks up treatment in matrix (§5.4)
    5. Determines escalation path (§5.5)
    6. Checks arrangement eligibility (§5.6)
    7. Assesses capacity constraints (§5.9)
    8. Records full evidence (§5.15)

    Returns dict with decision, bands, scores, and reason codes.
    """
    request = TreatmentRequest(
        account_id=account_id,
        client_id=client_id,
        decision_date=decision_date,
        days_past_due=days_past_due,
        arrears_amount=arrears_amount,
        balance=balance,
        product_family_code=product_family_code,
        contractual_instalment=contractual_instalment,
        times_cured_12m=times_cured_12m,
        times_cured_24m=times_cured_24m,
        right_party_contact_rate=right_party_contact_rate,
        promise_kept_rate=promise_kept_rate,
        broken_promises_90d=broken_promises_90d,
        age_months=age_months,
        other_accounts_in_arrears=other_accounts_in_arrears,
        sms_attempts_90d=sms_attempts_90d,
        call_attempts_90d=call_attempts_90d,
        email_attempts_90d=email_attempts_90d,
        successful_contacts_90d=successful_contacts_90d,
        last_payment_date=last_payment_date,
        last_payment_amount=last_payment_amount,
        payments_30d=payments_30d,
        payments_60d=payments_60d,
        payments_90d=payments_90d,
        active_arrangement=active_arrangement,
        arrangement_id=arrangement_id,
        consecutive_failed_arrangements=consecutive_failed_arrangements,
        debt_review_status_code=debt_review_status_code,
        hardship_arrangement=hardship_arrangement,
        deceased=deceased,
        external_debt_counselling=external_debt_counselling,
        dispute_raised=dispute_raised,
        litigation_in_progress=litigation_in_progress,
        episode_id=episode_id,
        previous_treatment_code=previous_treatment_code,
        previous_treatment_date=previous_treatment_date,
        channel_attempt_count=channel_attempt_count,
        intensity_ceiling=intensity_ceiling,
    )

    result = assess_treatment(request)

    return {
        # Decision
        "treatment_instance_id": result.treatment_instance_id,
        "decision_id": result.decision_id,
        "decision_date": str(result.decision_date),
        "treatment_code": result.treatment_code,
        "treatment_intensity": result.treatment_intensity,
        "channel_code": result.channel_code,

        # Bands and scores
        "arrears_bucket_code": result.arrears_bucket_code,
        "balance_band_code": result.balance_band_code,
        "collections_band_code": result.collections_band_code,
        "contact_band_code": result.contact_band_code,
        "collections_score": result.collections_score,
        "collections_score_unadjusted": result.collections_score_unadjusted,

        # Matrix attribution
        "matrix_version": result.matrix_version,
        "matrix_cell_id": result.matrix_cell_id,

        # Path
        "episode_id": result.episode_id,
        "path_position": result.path_position,
        "path_position_previous": result.path_position_previous,

        # Suspensions
        "active_suspensions": result.active_suspensions,
        "suspension_codes_applied": result.suspension_codes_applied,

        # Arrangements
        "arrangement_assessment_required": result.arrangement_assessment_required,
        "arrangement_id": result.arrangement_id,

        # Capacity
        "pool_name": result.pool_name,
        "rank_in_pool": result.rank_in_pool,
        "pool_size": result.pool_size,
        "non_selection_reason_code": result.non_selection_reason_code,

        # Overlays
        "adjustment_set_version": result.adjustment_set_version,
        "applied_overlay_ids": result.applied_overlay_ids,

        # Evidence
        "reason_codes": result.reason_codes,
    }


def build():
    """Build the collections treatment pipeline."""
    return flow(assign_treatment, name="collections_treatment")
