"""Business credit assessment, reusing project 05's components.

This module orchestrates the origination flow (EP-1) and annual review (EP-3/L1)
by importing and calling project 05's published capabilities without modification.

Reuse inventory:
- O2: Entity structure resolution → calls 05's structure resolver
- O3–O5: Entity assessment → calls 05's assess_business_credit()
- O6–O10: Blend and exposure → embedded in 05's output
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Optional, List, Dict, Any


@dataclass
class Facility:
    """Minimal facility record for annual review."""
    facility_id: int
    product_code: int  # 50=term, 51=revolving
    client_id: int
    amount: float
    term_months: int
    start_date: date
    review_date: date
    status_code: int  # 1=active, 2=delinquent, 3=restructured


@dataclass
class PriorDecision:
    """Prior decision of record for L1 comparability."""
    decision_id: int
    decision_date: date
    business_grade: int
    master_scale_version: str
    overlay_stack_unadjusted_grade: int


@dataclass
class OriginationResult:
    """Result of origination (EP-1) or review (EP-3/L1).

    Reused from 05 via passthrough: nearly all fields are 05's output.
    """
    application_id: int
    decision_date: date
    business_grade: int
    master_scale_version: str
    business_verdict_code: int  # 1=clear, 2=refer, 3=decline
    declined_entity_key: Optional[str]
    declined_rule: Optional[str]

    # New for L1
    prior_decision_id: Optional[int] = None
    grade_migration: Optional[int] = None  # New grade - prior grade
    grade_change_reason_codes: List[str] = None  # causes of migration


def assess_origination(
    application_id: int,
    decision_date: date,
    product_code: int,
    entities_list: List[Dict[str, Any]],
    adverse_events_list: List[Dict[str, Any]],
    assess_business_credit_fn=None,  # imported from 05
    knowledge_date: Optional[date] = None,
) -> OriginationResult:
    """Origination (EP-1) by calling project 05's assess_business_credit.

    This is a transparent pass-through to project 05 §5.1–5.13.
    No new logic; 05 is consumed as-is.
    """
    if knowledge_date is None:
        knowledge_date = decision_date

    # Call 05's capability
    if assess_business_credit_fn is None:
        # Stub for testing
        return _stub_origination(application_id, decision_date, product_code)

    result = assess_business_credit_fn(
        application_id=application_id,
        decision_date=decision_date,
        entities_list=entities_list,
        adverse_events_list=adverse_events_list,
        knowledge_date=knowledge_date,
    )

    return OriginationResult(
        application_id=application_id,
        decision_date=decision_date,
        business_grade=result.get("business_grade", 6),
        master_scale_version=result.get("master_scale_version", "v1.0"),
        business_verdict_code=result.get("business_verdict_code", 1),
        declined_entity_key=result.get("declined_entity_key"),
        declined_rule=result.get("declined_rule"),
    )


def assess_annual_review(
    facility_id: int,
    client_id: int,
    decision_date: date,
    review_date: date,
    entities_list: List[Dict[str, Any]],
    adverse_events_list: List[Dict[str, Any]],
    prior_decision: Optional[PriorDecision],
    assess_business_credit_fn=None,  # from 05
) -> OriginationResult:
    """Annual review (EP-3/L1) by comparing against prior decision.

    §5.10 Comparability: Compare this year's grade against prior decision's grade,
    decomposing the movement into:
    1. Data changed (entities, events, financials)
    2. Model changed (scorecard recalibration)
    3. Overlay changed (policy adjustments)
    4. Scale changed (master scale restatement)

    This implementation stubs the decomposition; full version in 09 swap-set.
    """
    knowledge_date = review_date

    # Re-assess using 05's component (same as origination)
    current = assess_origination(
        application_id=facility_id,
        decision_date=decision_date,
        product_code=50,  # stub
        entities_list=entities_list,
        adverse_events_list=adverse_events_list,
        assess_business_credit_fn=assess_business_credit_fn,
        knowledge_date=knowledge_date,
    )

    if prior_decision is None:
        # First review, nothing to compare
        return current

    # Grade migration (§5.10)
    current.prior_decision_id = prior_decision.decision_id
    current.grade_migration = current.business_grade - prior_decision.business_grade

    # Decompose movement (simplified)
    # Full version would compare:
    # - Unadjusted grades (scorecard + calibration only)
    # - Overlays applied (policy stack effect)
    # - Scale restatement (if master_scale_version differs)
    if current.master_scale_version != prior_decision.master_scale_version:
        current.grade_change_reason_codes = ["MASTER_SCALE_CHANGE"]
    elif current.grade_migration != 0:
        current.grade_change_reason_codes = ["DATA_OR_MODEL_CHANGE"]
    else:
        current.grade_change_reason_codes = []

    return current


def covenant_dscr_test(
    facility_id: int,
    test_date: date,
    ebitda: float,
    debt_service_annual: float,
    dscr_threshold: float = 1.25,
    definition_version: str = "2026-01",
) -> Dict[str, Any]:
    """Single covenant test: DSCR ≥ 1.25.

    §5.5 Covenants: One covenant instance, pinned to definition version.
    The definition version is frozen when the facility is documented, not
    updated by policy change.

    Returns result including the definition version used, for replay.
    """
    if debt_service_annual <= 0:
        return {
            "facility_id": facility_id,
            "test_date": test_date,
            "covenant_type": "DSCR",
            "definition_version": definition_version,
            "breach_class_code": 4,  # severe
            "reason": "No debt service",
        }

    dscr = ebitda / debt_service_annual

    if dscr < dscr_threshold:
        breach_class = 2  # material
        if dscr < dscr_threshold * 0.8:
            breach_class = 4  # severe
    else:
        breach_class = 0  # none

    return {
        "facility_id": facility_id,
        "test_date": test_date,
        "covenant_type": "DSCR",
        "definition_version": definition_version,
        "measured_value": dscr,
        "threshold": dscr_threshold,
        "breach_class_code": breach_class,
        "headroom_pct": ((dscr / dscr_threshold) - 1) * 100 if dscr > 0 else -100,
    }


def _stub_origination(
    application_id: int,
    decision_date: date,
    product_code: int,
) -> OriginationResult:
    """Stub origination when 05 not available."""
    return OriginationResult(
        application_id=application_id,
        decision_date=decision_date,
        business_grade=6,
        master_scale_version="v1.0",
        business_verdict_code=1,
        declined_entity_key=None,
        declined_rule=None,
    )
