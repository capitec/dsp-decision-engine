"""Collections assessment data models."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List


@dataclass
class TreatmentRequest:
    """Input to the collections treatment assignment flow."""
    # Account identity
    account_id: int
    client_id: int
    decision_date: date

    # Delinquency state (as at decision_date)
    days_past_due: int
    arrears_amount: float
    balance: float

    # Account characteristics
    product_family_code: int  # 1=unsecured term, 2=revolving, 3=secured asset, 4=business
    contractual_instalment: float

    # Score inputs (28 characteristics)
    times_cured_12m: int
    times_cured_24m: int
    right_party_contact_rate: float
    promise_kept_rate: float
    broken_promises_90d: int
    age_months: int
    other_accounts_in_arrears: int

    # Contact history (90 days)
    sms_attempts_90d: int
    call_attempts_90d: int
    email_attempts_90d: int
    successful_contacts_90d: int

    # Payment history
    last_payment_date: Optional[date] = None
    last_payment_amount: float = 0.0
    payments_30d: int = 0
    payments_60d: int = 0
    payments_90d: int = 0

    # Arrangement history
    active_arrangement: bool = False
    arrangement_id: Optional[int] = None
    consecutive_failed_arrangements: int = 0

    # Status
    debt_review_status_code: Optional[int] = None  # 101-104
    hardship_arrangement: bool = False
    deceased: bool = False
    external_debt_counselling: bool = False
    dispute_raised: bool = False
    litigation_in_progress: bool = False

    # Episode tracking
    episode_id: int = 0
    previous_treatment_code: Optional[int] = None
    previous_treatment_date: Optional[date] = None
    channel_attempt_count: int = 0  # per channel
    intensity_ceiling: int = 1

    # Cohort assignment (deterministic from account_id + salt)
    cohort_code: int = field(default=0, init=False)


@dataclass
class TreatmentResult:
    """Output of the collections treatment assignment."""
    # Decision identity
    treatment_instance_id: int
    decision_id: int
    decision_date: date

    # Assigned treatment
    treatment_code: int  # 0-13
    treatment_intensity: int  # 1-5
    channel_code: Optional[int]  # 1=SMS, 2=Email, 3=In-app, 4=IVM, 5=Voice call, 6=Field

    # Account state
    arrears_bucket_code: int  # 1-8
    balance_band_code: int  # 1-7
    collections_band_code: int  # 1-6
    contact_band_code: int  # 1-4
    collections_score: float
    collections_score_unadjusted: float

    # Matrix attribution
    matrix_version: int
    matrix_cell_id: int

    # Path and escalation
    episode_id: int
    path_position: int
    path_position_previous: int

    # Suspensions
    active_suspensions: List[int] = field(default_factory=list)
    suspension_codes_applied: List[int] = field(default_factory=list)

    # Arrangements
    arrangement_assessment_required: bool = False
    arrangement_id: Optional[int] = None

    # Capacity allocation
    pool_name: Optional[str] = None
    rank_in_pool: Optional[int] = None
    pool_size: Optional[int] = None
    non_selection_reason_code: Optional[int] = None  # 200-270

    # Overlays
    adjustment_set_version: int = 0
    applied_overlay_ids: List[int] = field(default_factory=list)

    # Evidence
    reason_codes: List[int] = field(default_factory=list)
