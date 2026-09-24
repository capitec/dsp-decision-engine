"""Shared intermediates and decision state across the 18-phase flow.

This module defines the registry of values that pass between phases,
implementing 10 §5.21: 41 values produced in one phase, consumed in multiple.
One dual-versioned value (existing_obligations in hypothetical scenarios)
demonstrates the hard case of §5.21.1.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import date


@dataclass
class ScenarioObligation:
    """One scenario's obligations view (§5.21.1)."""
    scenario_ref: Optional[int]  # null for actual; 1..250 for hypothetical
    existing_obligations: float  # Monthly cost of existing debt
    revolving_utilisation: float  # Existing revolving utilisation
    worst_arrears_months: int
    total_exposure: float
    discretionary_income: float
    max_affordable_instalment: float
    affordability_verdict_code: int


@dataclass
class DecisionState:
    """Shared intermediates persisted across phases for one decision.

    §5.31: Single decision record shape across all entry points.
    Every value names its basis (value_basis_code, scenario_ref per §5.21.1).
    """
    # P01: Request validation and routing
    entry_point_code: int = 1  # 1..8
    decision_date: Optional[date] = None  # Fixed once, never re-read (O-04)
    phase_set_id: str = "ep1_product10"  # e.g. "ep1_product10" – determines which phases run
    product_codes: List[int] = field(default_factory=list)  # Candidate products from request
    degradation_mode: Optional[str] = None  # e.g. "identity_service_down"
    loop_pass_count: int = 0  # L1: how many times has affordability rerun? (0=first pass)

    # P02: Client and identity resolution
    client_id: Optional[str] = None
    identity_confidence: float = 0.0
    resolution_path_code: Optional[int] = None  # R1..R4
    identity_degraded: bool = False  # True if identity_verification_service down
    related_party_ids: List[str] = field(default_factory=list)

    # P03: Consent and hard eligibility
    consent_state: Dict[str, bool] = field(default_factory=dict)  # channel -> has_consent
    hard_eligibility_pass: bool = False

    # P04: Data acquisition orchestration
    bureau_as_of_date: Optional[date] = None
    bureau_is_stale: bool = False
    external_call_budget_used_ms: float = 0.0

    # P05: Fraud and financial crime
    fraud_verdict_code: int = 1  # 1=pass, 2=soft decline, 3=hard decline, 4=refer
    fraud_reason_codes: List[str] = field(default_factory=list)

    # P06: Feature derivation
    gross_monthly_income: float = 0.0
    net_monthly_income: float = 0.0  # Consumed by P07..P18 (11 consumers)
    living_expenses: float = 0.0
    existing_obligations: float = 0.0  # Actual version
    worst_arrears_months: int = 0
    total_exposure: float = 0.0
    segment_code: int = 0  # Consumed by P07..P16 (7 consumers)
    income_band_code: int = 0

    # P08: Scoring and calibration/grading
    score: float = 0.0
    probability_of_default: float = 0.0  # 5 consumers (P09, P12, P15, P16, P17)
    risk_grade: int = 0  # 9 consumers (P09, P10, P11, P12, P13, P14, P15, P16, P17)
    adjustment_set_id: str = ""  # 9 consumers (P09, P10, P12, P13, P14, P15, P16, P17, P18)

    # P09: Policy gates and cap waterfall
    amount_cap: float = 0.0
    term_cap: int = 0
    worst_acceptable_grade: int = 0

    # P10: Affordability
    max_affordable_instalment: float = 0.0  # Shared intermediate to P12..P17
    affordability_verdict_code: int = 0  # 1=pass, 2=fail, 3=indeterminate
    affordability_pass: bool = False

    # P11: Product routing
    routed_product_code: int = 0  # The chosen product for this decision

    # P12: Pricing
    nominal_annual_rate: float = 0.0
    initiation_fee: float = 0.0
    monthly_service_fee: float = 0.0
    instalment: float = 0.0  # 6 consumers

    # P13: The solve (circular amount<->rate<->instalment)
    proposed_amount: float = 0.0
    proposed_term_months: int = 0
    solve_binding_constraint: str = ""  # which constraint limited the offer?

    # P14: Consolidation search (conditional on P10 failure + eligibility)
    consolidation_eligible: bool = False
    consolidation_scenarios_evaluated: int = 0
    chosen_scenario_ref: Optional[int] = None  # null for actual, 1..250 for hypothetical

    # P15: Limit assignment (not in entry point 1; entry point 2 only)
    proposed_limit: float = 0.0

    # P16: Offer assembly
    offers: List[Dict] = field(default_factory=list)
    offers_assembled: int = 0

    # P17: Final validation
    validation_pass: bool = False
    validation_failures: List[str] = field(default_factory=list)

    # P18: Decision record emission
    outcome_code: int = 0  # 1=approved, 2=decline, 3=refer, etc.
    reason_codes: List[str] = field(default_factory=list)
    decision_id: str = ""

    # Versions for replay (09 §5.15)
    rate_card_version: str = "1.0.0"
    table_versions_used: Dict[str, str] = field(default_factory=dict)
