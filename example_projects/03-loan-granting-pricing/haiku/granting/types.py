"""Data types for unsecured loan granting and pricing."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import date


class EligibilityGateCode(str, Enum):
    """Eligibility gate identifiers."""
    PRODUCT_CHANNEL = "1102"
    MIN_AGE = "1110"
    MAX_AGE = "1111"
    CAPACITY = "1112"
    RESIDENCY = "1120"
    EMPLOYMENT = "1130"
    EMPLOYMENT_PENSION = "1131"
    DEBT_REVIEW = "1140"
    ADMIN_ORDER = "1141"
    INSOLVENCY = "1142"
    DECEASED = "1150"
    EXCLUSION = "1160"
    DUPLICATE = "1170"
    IN_FLIGHT = "1171"


class CapType(str, Enum):
    """The three ceilings in the cap waterfall."""
    AMOUNT = "amount_cap"
    TERM = "term_cap"
    GRADE = "worst_acceptable_grade"


class OfferSuppressionReasonCode(str, Enum):
    """Why an offer term was suppressed."""
    MIN_AMOUNT = "1410"
    MIN_INSTALMENT = "1411"
    TOTAL_COST_RATIO = "1412"
    SCHEDULED_IN_DUPLUM = "1413"
    EAR_CEILING = "1414"
    TERM_CAP = "1415"
    EVAL_CEILING = "1420"


class ConstraintCode(str, Enum):
    """What constraint bound the solve for a term."""
    AFFORDABILITY = "BIND-AFF"
    CAP = "BIND-CAP"
    REQUESTED = "BIND-REQ"
    MINIMUM = "BIND-MIN"
    MAXIMUM = "BIND-MAX"
    TOTAL_COST_RATIO = "BIND-TCR"
    SCHEDULED_IN_DUPLUM = "BIND-DUP"
    RATE_CEILING = "BIND-CEIL"
    EVAL_CEILING = "BIND-EXH"


class OutcomeCode(str, Enum):
    """Final decision outcome."""
    APPROVE = "approve"
    APPROVE_WITH_CONDITIONS = "approve_with_conditions"
    REFER = "refer"
    DECLINE = "decline"


@dataclass
class GateVerdict:
    """Per-gate eligibility verdict."""
    gate_code: str
    gate_name: str
    passed: bool
    evaluated: bool  # False if data not yet available
    reason_code: Optional[str] = None
    input_value: Optional[Any] = None
    notes: str = ""


@dataclass
class CapChainLink:
    """One step in a ceiling's chain."""
    sequence: int
    rule_id: str
    rule_name: str
    applicable: bool
    value_before: float
    value_after: float
    status: str  # "seed", "bound", "evaluated_not_bound", "not_applicable", "raised"


@dataclass
class PricingEvaluation:
    """One pricing evaluation in the solve."""
    candidate_amount: float
    rate_cell_id: str
    nominal_rate: float
    advance: float
    financed: float
    initiation_fee: float
    monthly_service_fee: float
    credit_life_premium: float
    instalment: float
    affordable: bool
    reason_if_not: Optional[str] = None


@dataclass
class TermResult:
    """Result of solve for one term."""
    term_months: int
    affordable_amount: Optional[float]
    binding_constraint: Optional[str]
    evaluation_count: int
    evaluations: List[PricingEvaluation] = field(default_factory=list)
    suppression_reasons: List[str] = field(default_factory=list)
    pricing: Optional[Dict[str, Any]] = None  # If an offer was made


@dataclass
class OfferOutput:
    """One offer in the final offer set."""
    term_months: int
    offered_amount: float
    rate_cell_id: str
    nominal_annual_rate: float
    initiation_fee: float
    monthly_service_fee: float
    credit_life_premium: float
    instalment: float
    total_cost_of_credit: float
    effective_annual_rate: float
    total_cost_ratio: float
    is_recommended: bool = False
    binding_constraint: str = ""


@dataclass
class GrantingResult:
    """Complete granting decision output."""
    decision_id: str
    decision_date: date
    application_id: int
    client_id: int

    # Gates
    eligibility_verdicts: List[GateVerdict]
    is_eligible: bool

    # Risk and affordability
    fraud_verdict_code: int
    scorecard_id: str
    score: float
    score_unadjusted: float
    probability_of_default: float
    probability_of_default_unadjusted: float
    risk_grade: int
    risk_grade_unadjusted: int
    max_affordable_instalment: float

    # Caps
    amount_cap: float
    term_cap: int
    worst_acceptable_grade: int

    # Outcome
    outcome_code: str
    reason_codes: List[str]

    # Optional fields
    primary_reason_code: Optional[str] = None
    referral_queue_code: Optional[str] = None
    cap_chains: Dict[str, List[CapChainLink]] = field(default_factory=dict)
    offers: List[OfferOutput] = field(default_factory=list)
    recommended_offer: Optional[OfferOutput] = None
    table_versions: Dict[str, str] = field(default_factory=dict)
    overlay_stack_id: Optional[str] = None
    adjustments_applied: List[Dict[str, Any]] = field(default_factory=list)
