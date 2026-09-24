"""Collections treatment assessment logic (project 08 §5)."""
from __future__ import annotations
from datetime import date, timedelta
from typing import Optional, List, Tuple
import hashlib
from .model import TreatmentRequest, TreatmentResult


def compute_arrears_bucket(days_past_due: int, buckets: List[Tuple[int, int]]) -> int:
    """Map days_past_due to bucket code (1-8)."""
    for i, (lo, hi) in enumerate(buckets, 1):
        if lo <= days_past_due <= hi:
            return i
    return 8  # 365+


def compute_balance_band(balance: float, bands: List[float]) -> int:
    """Map balance to band code (1-7)."""
    for i, threshold in enumerate(bands, 1):
        if balance < threshold:
            return i
    return 7  # >= 250k


def compute_collections_score(request: TreatmentRequest) -> Tuple[float, float]:
    """
    Simplified collections score from 28 characteristics.
    Returns (score, score_unadjusted).

    ponytail: Real model has 28 characteristics with null bins; this uses ~10 key ones.
    """
    # Base score from bucket
    score = 40.0 + (request.days_past_due / 100.0)

    # Payment history
    if request.payments_90d > 0:
        score -= 15.0
    if request.last_payment_date and isinstance(request.last_payment_date, date):
        days_since_payment = (request.decision_date - request.last_payment_date).days
        if days_since_payment < 7:
            score -= 10.0
        elif days_since_payment < 30:
            score -= 5.0

    # Contact responsiveness
    if request.successful_contacts_90d > 0:
        score -= 12.0
    if request.right_party_contact_rate > 0.5:
        score -= 8.0

    # Promise history
    if request.promise_kept_rate > 0.8:
        score -= 10.0
    if request.broken_promises_90d > 2:
        score += 10.0

    # Cure history
    if request.times_cured_12m > 0:
        score -= 20.0
    if request.times_cured_24m > 1:
        score -= 15.0

    # Account characteristics
    if request.other_accounts_in_arrears > 1:
        score += 5.0
    if request.age_months > 60:
        score -= 5.0

    # Clamp to 0-100
    score = max(0.0, min(100.0, score))

    return score, score


def band_collections_score(score: float, band_edges: List[float]) -> int:
    """Map collections score to band code (1-6)."""
    for i, edge in enumerate(band_edges, 1):
        if score < edge:
            return i
    return 6


def compute_contact_band(
    successful_contacts_90d: int,
    attempts_per_contact: float,
    valid_channels: int
) -> int:
    """Compute contact band (1=responsive, 4=unreachable)."""
    if successful_contacts_90d == 0:
        return 4  # unreachable
    if attempts_per_contact > 3.0:
        return 3  # silent
    if successful_contacts_90d > 5 or attempts_per_contact < 1.5:
        return 1  # responsive
    return 2  # intermittent


def evaluate_suspensions(request: TreatmentRequest) -> Tuple[List[int], bool]:
    """
    Evaluate all active suspensions (§5.2).
    Returns (list of suspension codes, whether any block all contact).

    ponytail: Implements 20 codes; production tracks expiry per code.
    """
    suspensions: List[int] = []
    blocks_all_contact = False

    # 101-104: Debt review
    if request.debt_review_status_code in (101, 102, 103):
        suspensions.append(request.debt_review_status_code)
        if request.debt_review_status_code in (101, 102, 103):
            blocks_all_contact = True

    # 111: Hardship arrangement
    if request.hardship_arrangement:
        suspensions.append(111)
        blocks_all_contact = True

    # 107: Deceased
    if request.deceased:
        suspensions.append(107)
        blocks_all_contact = True

    # 108: Internal complaint (stub)
    if request.external_debt_counselling:
        suspensions.append(108)

    # 110: Dispute
    if request.dispute_raised:
        suspensions.append(110)
        blocks_all_contact = True

    # 113: Litigation
    if request.litigation_in_progress:
        suspensions.append(113)

    return suspensions, blocks_all_contact


def lookup_treatment_matrix(
    bucket: int,
    collections_band: int,
    balance_band: int,
    contact_band: int,
    product_family: int,
    matrix: dict
) -> Tuple[int, int, int, int, int]:
    """
    Look up treatment in the 5-dimension matrix.
    Returns (treatment_code, intensity, retries, cooling_off, cell_id).

    ponytail: Generated sparse matrix; production is 5376 cells in structured form.
    """
    key = (bucket, collections_band, balance_band, contact_band, product_family)
    cell = matrix.get(key)
    if cell is None:
        # No explicit cell = no treatment
        cell_id = hash(key) % 5376
        return 0, 1, 0, 0, cell_id

    cell_id = cell.get("cell_id", hash(key) % 5376)
    return (
        cell.get("treatment_code", 0),
        cell.get("intensity", 1),
        cell.get("permitted_retries", 0),
        cell.get("cooling_off_days", 0),
        cell_id
    )


def compute_episode_and_position(
    request: TreatmentRequest,
    treatment_from_matrix: int
) -> Tuple[int, int]:
    """
    Determine episode and path position (§5.5).
    Returns (episode_id, path_position).

    ponytail: Simplified; production tracks full history and reset events.
    """
    # Episode: use provided, or start new (day 1)
    episode_id = request.episode_id or hash((request.account_id, request.decision_date)) % 2**31

    # Position advancement: escalate if previous treatment failed and retries exhausted
    path_position = request.intensity_ceiling
    if request.previous_treatment_code is not None and request.previous_treatment_date and isinstance(request.previous_treatment_date, date):
        days_since = (request.decision_date - request.previous_treatment_date).days
        if days_since >= 1:  # simplified retry window
            path_position = min(path_position + 1, 5)

    return episode_id, path_position


def calculate_treatment_instance_id(
    account_id: int, decision_date: date, decision_sequence: int
) -> int:
    """Stable treatment instance identifier."""
    h = hashlib.sha256(
        f"{account_id}:{decision_date}:{decision_sequence}".encode()
    ).digest()
    return int.from_bytes(h[:8], "big", signed=False) % (2**63)


def assess_treatment(request: TreatmentRequest) -> TreatmentResult:
    """
    Main collections treatment assessment (§5.1-§5.9).

    Returns full evidence record per §5.15.
    """
    # ===== Stage 1: Account state assembly (§5.1) =====
    bucket_boundaries = [(0, 14), (15, 29), (30, 59), (60, 89), (90, 119), (120, 179), (180, 364), (365, 10000)]
    arrears_bucket = compute_arrears_bucket(request.days_past_due, bucket_boundaries)

    balance_band_edges = [2500, 10000, 25000, 50000, 100000, 250000, float('inf')]
    balance_band = compute_balance_band(request.balance, balance_band_edges)

    # ===== Stage 2: Suspensions (§5.2) =====
    active_suspensions, blocks_all = evaluate_suspensions(request)

    # ===== Stage 3: Risk assessment (§5.3) =====
    collections_score, score_unadjusted = compute_collections_score(request)
    band_edges = [20, 35, 50, 65, 80]
    collections_band = band_collections_score(collections_score, band_edges)
    contact_band = compute_contact_band(
        request.successful_contacts_90d,
        request.sms_attempts_90d / max(request.successful_contacts_90d, 1),
        3  # valid channels
    )

    # ===== Stage 4: Treatment matrix (§5.4) =====
    treatment_matrix = generate_sparse_matrix(arrears_bucket, collections_band, request.product_family_code)
    treatment_code, intensity, retries, cooling_off, cell_id = lookup_treatment_matrix(
        arrears_bucket, collections_band, balance_band, contact_band, request.product_family_code,
        treatment_matrix
    )

    # If suspended, check if treatment is blocked
    if blocks_all and treatment_code > 0:
        # Suspensions override
        treatment_code = 0
        intensity = 1
        non_selection_reason = 210  # Suspended

    # ===== Stage 5: Escalation path (§5.5) =====
    episode_id, path_position = compute_episode_and_position(request, treatment_code)

    # ===== Stage 6: Arrangements (§5.6) =====
    arrangement_required = treatment_code == 12 and not request.active_arrangement

    # ===== Stage 9: Capacity allocation (§5.9) =====
    # Stub: 9 pools, account gets ranked
    pool_name = None
    rank_in_pool = None
    pool_size = None
    non_selection_reason = None

    if treatment_code == 0:
        non_selection_reason = 200  # Matrix recommended no action
    elif blocks_all and treatment_code > 0:
        non_selection_reason = 210  # Suspended

    if treatment_code > 0 and treatment_code in (5, 6, 7):  # Agent calls
        pool_name = "early_agents" if arrears_bucket < 4 else "late_agents"
        rank_in_pool = (hash((request.account_id, request.decision_date)) % 100) + 1
        pool_size = 27000 if pool_name == "early_agents" else 11000

    # Deterministic cohort assignment
    cohort_salt = hash("cohort_experiment_1") & 0xFFFFFFFF
    cohort_code = ((request.account_id ^ cohort_salt) % 100) + 1

    # ===== Stage 10: Output assembly (§5.10) =====
    treatment_instance_id = calculate_treatment_instance_id(request.account_id, request.decision_date, 0)
    decision_id = hash((treatment_instance_id, request.decision_date)) % (2**63)

    return TreatmentResult(
        treatment_instance_id=treatment_instance_id,
        decision_id=decision_id,
        decision_date=request.decision_date,
        treatment_code=treatment_code,
        treatment_intensity=intensity,
        channel_code=map_treatment_to_channel(treatment_code),
        arrears_bucket_code=arrears_bucket,
        balance_band_code=balance_band,
        collections_band_code=collections_band,
        contact_band_code=contact_band,
        collections_score=round(collections_score, 2),
        collections_score_unadjusted=round(score_unadjusted, 2),
        matrix_version=1,
        matrix_cell_id=cell_id,
        episode_id=episode_id,
        path_position=path_position,
        path_position_previous=request.intensity_ceiling,
        active_suspensions=active_suspensions,
        suspension_codes_applied=active_suspensions if blocks_all else [],
        arrangement_assessment_required=arrangement_required,
        arrangement_id=request.arrangement_id,
        pool_name=pool_name,
        rank_in_pool=rank_in_pool,
        pool_size=pool_size,
        non_selection_reason_code=non_selection_reason,
        adjustment_set_version=0,
        applied_overlay_ids=[],
        reason_codes=[200] if treatment_code == 0 else active_suspensions if blocks_all else []
    )


def map_treatment_to_channel(treatment_code: int) -> Optional[int]:
    """Map treatment code to channel code."""
    channel_map = {
        1: 1,   # SMS
        2: 2,   # Email
        3: 3,   # In-app
        4: 4,   # IVM
        5: 5,   # Voice call
        6: 5,   # Voice call
        7: 5,   # Voice call
        8: 6,   # Field
        9: 7,   # Agency
        10: 2,  # Notice (email)
        11: 7,  # Legal (agency)
        12: 5,  # Settlement offer (voice)
        13: None  # Write-off (no channel)
    }
    return channel_map.get(treatment_code)


def generate_sparse_matrix(bucket: int, band: int, product: int) -> dict:
    """
    Generate a sparse treatment matrix.

    ponytail: Deterministic generation; production loads from structured spreadsheet.
    """
    matrix = {}

    # Simple rule: early buckets get automated treatments, late buckets get agent calls
    for b in range(1, 9):
        for cb in range(1, 7):
            for bb in range(1, 8):
                for contact_b in range(1, 5):
                    for prod in range(1, 5):
                        treatment_code = 0

                        # Escalation by bucket
                        if b <= 2:
                            if cb <= 3:
                                treatment_code = 1 if cb <= 2 else 2  # SMS or email
                            else:
                                treatment_code = 4  # IVM
                        elif b <= 4:
                            if cb <= 2:
                                treatment_code = 1  # SMS
                            elif cb <= 4:
                                treatment_code = 5  # Agent call
                            else:
                                treatment_code = 4
                        else:
                            treatment_code = 6 if cb <= 3 else 7  # Standard or high intensity

                        if treatment_code > 0:
                            cell_key = (b, cb, bb, contact_b, prod)
                            matrix[cell_key] = {
                                "cell_id": hash(cell_key) % 5376,
                                "treatment_code": treatment_code,
                                "intensity": min(1 + (b // 2), 5),
                                "permitted_retries": 3,
                                "cooling_off_days": 3 if treatment_code <= 4 else 1
                            }

    return matrix
