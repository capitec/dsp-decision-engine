"""Eighteen decision phases (P01-P18) for retail credit end-to-end flow.

Entry point 1 (new credit application) runs P01-P13, P16-P18, with P14 conditional.
Other entry points stub out or run subsets.
Each phase updates the shared DecisionState.
"""
from datetime import date, datetime, timedelta
from typing import Optional
import sys
import os

# Add core library to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/00-shared-credit-core")

from flow_state import DecisionState, ScenarioObligation
from decider import param


# ============================================================================
# P01: Request validation and routing
# ============================================================================

def p01_request_validation_and_routing(
    entry_point_code: int,
    product_code: int = param(10, ge=10, le=40),
    amount: float = param(50000, ge=2000, le=500000),
    term_months: int = param(36, ge=6, le=84),
    channel_code: int = param(1, ge=1, le=10),
    client_id_from_request: Optional[str] = None,
    consolidation_eligible: bool = param(False),
    decision_date_override: Optional[date] = None
) -> dict:
    """P01: Request validation and routing (41 decision points).

    §5.2: Validates structural completeness (14 checks), field domains (11),
    cross-field consistency (9). Routes to phase set.
    §5.20: Routes based on entry_point_code, product_code, channel, consolidation_eligible.
    O-04: decision_date fixed here, never re-read.
    """
    state = DecisionState(
        entry_point_code=entry_point_code,
        decision_date=decision_date_override or date.today(),
        product_codes=[product_code],
        consolidation_eligible=consolidation_eligible
    )

    # Determine phase_set based on entry_point_code
    # Entry point 1: P01-P13, P16-P18; P14 conditional
    phase_set_map = {
        1: "ep1_product10",  # New credit application (16 phases + conditional P14)
        2: "ep2_limit_change",  # Client-initiated limit change
        3: "ep3_limit_programme",  # Limit programme evaluation
        4: "ep4_campaign_preapproval",  # Campaign pre-approval
        5: "ep5_consolidation",  # Consolidation assessment
        6: "ep6_reprice",  # Re-price / retention offer
        7: "ep7_quotation",  # Quotation only
        8: "ep8_simulation",  # What-if simulation
    }
    state.phase_set_id = phase_set_map.get(entry_point_code, "ep1_product10")

    # Validation: 34 checks, simplified for scope
    structural_valid = (amount > 0 and term_months > 0 and channel_code > 0)
    domain_valid = (amount >= 2000 and amount <= 500000 and term_months >= 6 and term_months <= 84)
    cross_field_valid = True  # Simplified; full has 9 checks per §5.2

    if not (structural_valid and domain_valid and cross_field_valid):
        return {
            "validation_pass": False,
            "rejection_reason": "INVALID_REQUEST",
            "state": state
        }

    return {
        "validation_pass": True,
        "phase_set_id": state.phase_set_id,
        "state": state
    }


# ============================================================================
# P02: Client and identity resolution
# ============================================================================

def p02_client_resolution(
    state: DecisionState,
    identity_number: Optional[str] = None,
    client_name: Optional[str] = None,
    date_of_birth: Optional[date] = None
) -> dict:
    """P02: Client and identity resolution (33 decision points).

    §5.3: Four resolution paths (R1-R4) with confidence.
    O-03: Before P03 (consent must use correct client_id).
    O-08: Before segment assignment (segment depends on thin-file status).
    Degradation: identity_verification_service down → R3 unavailable, floor rises to 0.94.
    """
    # Simplified: assign a client_id if identity_number provided
    if identity_number:
        # Simulate confidence: would be R1, R2, R3, or R4 in reality
        state.client_id = identity_number
        state.identity_confidence = 0.97  # Simulating R2
        state.resolution_path_code = 2
    else:
        # No resolution
        state.client_id = None
        state.identity_confidence = 0.0
        state.resolution_path_code = None

    # Simulate related-party set (expensive, computed once per O-11)
    if state.client_id:
        # In reality this would be a graph query; here we stub it
        state.related_party_ids = [state.client_id]  # Simplified: just the client

    # Degradation: if identity_verification_service down, mark degraded
    # (In reality this would be caught via external call failure)
    state.identity_degraded = False

    return {
        "client_id": state.client_id,
        "identity_confidence": state.identity_confidence,
        "identity_degraded": state.identity_degraded,
        "state": state
    }


# ============================================================================
# P03: Consent and hard eligibility
# ============================================================================

def p03_consent_and_eligibility(
    state: DecisionState,
    has_bureau_consent: bool = param(True),
    has_data_sharing_consent: bool = param(True)
) -> dict:
    """P03: Consent and hard eligibility (62 decision points).

    §5.4: Consent (19 decision points) + hard eligibility (43 decision points).
    O-02: Before P04 (bureau enquiry without consent is an offence).
    O-03: After P02 (uses client_id resolved above).
    §5.4.1: Reduced form on EP3/EP4 (pre-resolved identity).
    """
    # Consent state
    state.consent_state = {
        "bureau": has_bureau_consent,
        "data_sharing": has_data_sharing_consent,
        "marketing": param(False),  # Often false, user can override
    }

    # Hard eligibility: client must exist, have bureau consent, be in hard-eligible segment
    if state.client_id and has_bureau_consent:
        state.hard_eligibility_pass = True
    else:
        state.hard_eligibility_pass = False

    return {
        "consent_obtained": has_bureau_consent and has_data_sharing_consent,
        "hard_eligibility_pass": state.hard_eligibility_pass,
        "state": state
    }


# ============================================================================
# P04: Data acquisition orchestration
# ============================================================================

def p04_data_acquisition(
    state: DecisionState,
    bureau_enquiry_ms: float = param(150.0, ge=0, le=600),
    fraud_intel_ms: float = param(100.0, ge=0, le=350),
    bank_statements_ms: float = param(800.0, ge=0, le=1400)
) -> dict:
    """P04: Data acquisition orchestration (§5.5).

    O-06: On EP1/EP2 below R40k, fraud after bureau. On EP5/EP6, fraud before bureau (cycle).
    O-02: Only if P03 granted consent.
    Budget: 600ms for bureau, 350ms for fraud, 1400ms for statements.
    """
    # Simulate external calls with budget
    call_time_ms = 0.0

    if state.consent_state.get("bureau", False):
        # Bureau enquiry (simplified)
        call_time_ms += bureau_enquiry_ms
        state.bureau_as_of_date = date.today()
        # In reality, check if bureau data is > 40 days old
        state.bureau_is_stale = False

    state.external_call_budget_used_ms = call_time_ms

    # Stub fraud data and bank statements
    # (Real implementation would call external services)

    return {
        "bureau_enquiry_complete": state.bureau_as_of_date is not None,
        "external_call_ms": state.external_call_budget_used_ms,
        "state": state
    }


# ============================================================================
# P05: Fraud and financial crime
# ============================================================================

def p05_fraud_assessment(
    state: DecisionState,
    fraud_verdict_code: int = 1  # 1=pass, 2=soft, 3=hard, 4=refer
) -> dict:
    """P05: Fraud and financial crime (188 decision points for EP1).

    O-06: Cycle broken by entry_point_code: low-value before bureau, high-value after.
    §5.6.1: Five rule families, 620 live + shadow rules (simplified to param here).
    Returns fraud_verdict_code (consumed by P09, P11, P14, P16, P17, P18).
    """
    # Simplified: just record the verdict and reason codes
    state.fraud_verdict_code = fraud_verdict_code
    if fraud_verdict_code > 1:
        state.fraud_reason_codes = ["FR-VELOCITY" if fraud_verdict_code == 2 else "FR-SYNTHETIC"]

    return {
        "fraud_verdict_code": fraud_verdict_code,
        "fraud_reason_codes": state.fraud_reason_codes,
        "state": state
    }


# ============================================================================
# P06: Feature derivation
# ============================================================================

def p06_feature_derivation(
    state: DecisionState,
    gross_monthly_income: float = param(30000, ge=1000, le=200000),
    dependants: int = param(0, ge=0, le=10),
    employment_type_code: int = param(1, ge=1, le=6),
    bureau_accounts: int = param(3, ge=0, le=20),
    internal_tenure_months: int = param(24, ge=0, le=360)
) -> dict:
    """P06: Feature derivation (§5.7, 74 decision points).

    Computes: net_monthly_income (11 consumers), segment_code (7), living_expenses,
    existing_obligations, worst_arrears_months, total_exposure.
    O-07: Before P07 (14 scorecard characteristics are income-derived).
    O-08: Before segment assignment (thin-file status is bureau-derived).
    Returns intermediate values consumed by later phases.
    """
    # Gross income
    state.gross_monthly_income = gross_monthly_income

    # Deductions (simplified: use core.deductions if available)
    statutory_deductions = gross_monthly_income * 0.15  # Simplified
    state.net_monthly_income = gross_monthly_income - statutory_deductions

    # Living expenses (simplified: use core.expense_norms)
    norm_expense_floor = 2000.0 + (dependants * 500)  # Simplified
    stated_expenses = 3000.0
    state.living_expenses = max(norm_expense_floor, stated_expenses)

    # Existing obligations (actual version; hypothetical versions in P14 scenario loop)
    # §5.21.1: This is the "hard case" of two simultaneous versions
    state.existing_obligations = 5000.0  # Simplified; would de-dup bureau + internal

    # Segment assignment (12 segments; simplified to just thin vs thick file)
    if bureau_accounts < 3 or internal_tenure_months < 15:
        state.segment_code = 1  # Thin file
    else:
        state.segment_code = 4  # Existing, clean

    # Arrears and exposure (simplified)
    state.worst_arrears_months = 0
    state.total_exposure = state.existing_obligations * 5  # Rough approximation

    return {
        "net_monthly_income": state.net_monthly_income,
        "segment_code": state.segment_code,
        "existing_obligations": state.existing_obligations,
        "state": state
    }


# ============================================================================
# P07: Scoring (§5.8)
# ============================================================================

def p07_scoring(
    state: DecisionState,
    base_score: float = param(650, ge=300, le=850)
) -> dict:
    """P07: Scoring (§5.8, 61 decision points on scorecard A3).

    O-07: Runs after P06 (income-derived characteristics).
    Consumes: net_monthly_income, segment_code.
    Returns: score, probability_of_default (consumed by P09, P12, P15, P16, P17).
    """
    # Simplified: score is base + income adjustment
    income_adj = (state.net_monthly_income / 50000) * 50  # Rough scale
    state.score = base_score + income_adj

    # Calibrate score to PD using segment
    # In reality: use core.calibration
    if state.segment_code == 1:
        # Thin file, slightly higher default risk
        state.probability_of_default = min(0.05 + (state.score - 700) * 0.001, 0.15)
    else:
        state.probability_of_default = max(0.01 + (state.score - 750) * 0.001, 0.001)

    return {
        "score": state.score,
        "probability_of_default": state.probability_of_default,
        "state": state
    }


# ============================================================================
# P08: Calibration, grading and adjustments (§5.9)
# ============================================================================

def p08_calibration_and_grading(
    state: DecisionState,
    grade_override: Optional[int] = None
) -> dict:
    """P08: Calibration, grading and adjustments (196 decision points).

    O-01: Adjustments applied before grading (grade keys off adjusted PD).
    O-05: Resolve overlay register once, before phases that read overlaid values.
    Returns: risk_grade, adjustment_set_id (9 consumers each).
    """
    # Simplified: PD to grade mapping
    pd = state.probability_of_default
    if pd < 0.01:
        state.risk_grade = 1
    elif pd < 0.02:
        state.risk_grade = 2
    elif pd < 0.05:
        state.risk_grade = 3
    elif pd < 0.08:
        state.risk_grade = 4
    elif pd < 0.12:
        state.risk_grade = 5
    elif pd < 0.18:
        state.risk_grade = 6
    elif pd < 0.25:
        state.risk_grade = 7
    elif pd < 0.35:
        state.risk_grade = 8
    elif pd < 0.50:
        state.risk_grade = 9
    elif pd < 0.70:
        state.risk_grade = 10
    elif pd < 0.85:
        state.risk_grade = 11
    else:
        state.risk_grade = 12

    if grade_override:
        state.risk_grade = grade_override

    # Overlay register: which overlay set applies?
    # Simplified: just record a set_id
    state.adjustment_set_id = f"adjustments_seg{state.segment_code}_g{state.risk_grade}"

    return {
        "risk_grade": state.risk_grade,
        "adjustment_set_id": state.adjustment_set_id,
        "state": state
    }


# ============================================================================
# P09: Policy gates and cap waterfall (§5.10, 196 decision points)
# ============================================================================

def p09_policy_gates_and_caps(
    state: DecisionState,
    cap_rules_count: int = param(52, ge=1, le=100)  # Real: 52 rules
) -> dict:
    """P09: Policy gates and cap waterfall (196 decision points).

    O-09: Run before P10 with conservative buffer, then after P11 with routed product buffer.
    O-10: Amount/term entries before P10; instalment entries after.
    O-12: CAP-0118 uplift last (only entry that may raise ceiling).
    O-13: Statutory rate ceiling check after rate add-on.
    Returns: amount_cap, term_cap, worst_acceptable_grade (6 consumers each).
    """
    # Simplified: compute caps based on grade
    base_cap = 250000  # Base
    grade_factor = 1.0 - (state.risk_grade - 1) * 0.05  # Cap reduces with grade
    state.amount_cap = base_cap * max(grade_factor, 0.3)
    state.term_cap = 84
    state.worst_acceptable_grade = 8  # Can offer down to grade 8

    return {
        "amount_cap": state.amount_cap,
        "term_cap": state.term_cap,
        "worst_acceptable_grade": state.worst_acceptable_grade,
        "state": state
    }


# ============================================================================
# P10: Affordability (§5.11, §5.23 loop)
# ============================================================================

def p10_affordability_assessment(
    state: DecisionState,
    buffer_override: Optional[float] = None,
    force_fail: bool = False,
    max_loop_passes: int = param(4, ge=1, le=10)
) -> dict:
    """P10: Affordability assessment (71 decision points + loop L1).

    O-09: Cycle: run before P11 with conservative buffer, then re-run after P11 with product buffer.
    O-15: P14 sits inside the P10 loop (consolidation search runs when affordability fails).
    §5.23.1: Loop L1 re-runs P10, P13, P14 when P10 fails and client consolidation-eligible.
    Returns: max_affordable_instalment (6 consumers), affordability_verdict_code.
    """
    # Calculate discretionary income
    discretionary_income = state.net_monthly_income - state.living_expenses - state.existing_obligations

    # Apply affordability buffer
    buffer = buffer_override or 0.12
    max_affordable = discretionary_income * (1.0 - buffer)
    state.max_affordable_instalment = max(max_affordable, 0)

    # Determine verdict
    if force_fail or discretionary_income < 0:
        state.affordability_verdict_code = 2  # Fail
        state.affordability_pass = False
    else:
        state.affordability_verdict_code = 1  # Pass
        state.affordability_pass = True

    # Loop L1 logic: if affordability fails and consolidation-eligible, mark for P14
    should_loop = (
        state.affordability_verdict_code == 2 and
        state.consolidation_eligible and
        state.loop_pass_count < (max_loop_passes - 1)
    )

    return {
        "max_affordable_instalment": state.max_affordable_instalment,
        "affordability_verdict_code": state.affordability_verdict_code,
        "should_trigger_consolidation_loop": should_loop,
        "state": state
    }


# ============================================================================
# P11: Product routing (§5.12)
# ============================================================================

def p11_product_routing(
    state: DecisionState,
    request_product: int = param(10, ge=10, le=40)
) -> dict:
    """P11: Product routing (§5.12, 14 decision points).

    O-09: After P10 affordability with conservative buffer, then before P10 re-run with product buffer.
    Routes to one product based on request + eligibility.
    Entry point 1: product 10 only.
    """
    # For EP1, route to product 10 (Flex Loan)
    state.routed_product_code = request_product if request_product == 10 else 10

    return {
        "routed_product_code": state.routed_product_code,
        "state": state
    }


# ============================================================================
# P12: Pricing (§5.13, 16 decision points)
# ============================================================================

def p12_pricing(
    state: DecisionState,
    amount: float = param(50000, ge=2000, le=500000),
    term_months: int = param(36, ge=6, le=84),
    rate_override: Optional[float] = None
) -> dict:
    """P12: Pricing (16 decision points).

    O-14: Within phase: rate → fee → premium → instalment.
    Returns: nominal_annual_rate, instalment (5 consumers each).
    Consumes: risk_grade, amount_cap, max_affordable_instalment.
    """
    # Rate lookup: grade-based
    if rate_override:
        state.nominal_annual_rate = rate_override
    else:
        base_rate = 0.15  # 15%
        grade_spread = (state.risk_grade - 6) * 0.01  # Grade 6 @ 15%, 7 @ 16%, etc.
        state.nominal_annual_rate = base_rate + grade_spread

    # Fees (simplified)
    state.initiation_fee = amount * 0.01  # 1%
    state.monthly_service_fee = 50.0

    # Instalment calculation (simplified: uniform amortization)
    monthly_rate = state.nominal_annual_rate / 12
    numerator = amount + state.initiation_fee
    if monthly_rate > 0:
        state.instalment = (numerator * monthly_rate) / (1 - (1 + monthly_rate) ** (-term_months))
    else:
        state.instalment = numerator / term_months

    state.instalment += state.monthly_service_fee

    return {
        "nominal_annual_rate": state.nominal_annual_rate,
        "initiation_fee": state.initiation_fee,
        "instalment": state.instalment,
        "state": state
    }


# ============================================================================
# P13: The solve (§5.14, 28 decision points, circular amount↔rate↔instalment)
# ============================================================================

def p13_the_solve(
    state: DecisionState,
    requested_amount: float = param(50000, ge=2000, le=500000),
    requested_term: int = param(36, ge=6, le=84)
) -> dict:
    """P13: The solve (28 decision points).

    §5.14: Solves circular constraint: max affordable amount given rate, term, affordability.
    Returns: proposed_amount, proposed_term_months, solve_binding_constraint.
    Binding constraint: affordability | amount_cap | term_cap | grade_cap.
    """
    # Simplified solve: start with request, constrain by caps and affordability
    solved_amount = min(requested_amount, state.amount_cap)
    solved_term = min(requested_term, state.term_cap)

    # Re-price at solved amount/term to find binding constraint
    # (Real implementation iterates until instalment <= max_affordable)
    binding = "affordability"  # Simplified

    state.proposed_amount = solved_amount
    state.proposed_term_months = solved_term
    state.solve_binding_constraint = binding

    return {
        "proposed_amount": solved_amount,
        "proposed_term_months": solved_term,
        "solve_binding_constraint": binding,
        "state": state
    }


# ============================================================================
# P14: Consolidation search (§5.15, conditional on P10 failure)
# ============================================================================

def p14_consolidation_search(
    state: DecisionState,
    max_scenarios: int = param(5, ge=1, le=250)  # Real: up to 250
) -> dict:
    """P14: Consolidation search (§5.15).

    O-15: Sits inside P10 loop, not before/after (reached because affordability failed).
    §5.21.1: Hard case – two simultaneous versions of existing_obligations.
    Only runs if affordability failed and client is consolidation_eligible.
    Simplified: stub with scenario count, use actual version if no scenario chosen.
    """
    # For now, stub: no scenarios explored in this implementation
    # Real version would:
    # 1. Generate settlement scenarios
    # 2. Re-run affordability for each (hypothetical obligations)
    # 3. Select best scenario or null
    # 4. Record all rejected scenarios with rejection_reason_code

    state.consolidation_scenarios_evaluated = 0
    state.chosen_scenario_ref = None  # Null = use actual obligations

    return {
        "consolidation_scenarios_evaluated": state.consolidation_scenarios_evaluated,
        "chosen_scenario_ref": state.chosen_scenario_ref,
        "state": state
    }


# ============================================================================
# P15: Limit assignment (§5.16, not in EP1; EP2/EP3 only)
# ============================================================================

def p15_limit_assignment(
    state: DecisionState,
    proposed_limit: float = param(100000, ge=1000, le=300000)
) -> dict:
    """P15: Limit assignment (§5.16, 27 decision points).

    Not in Entry point 1 (term loan has no limit).
    Stub for routing.
    """
    state.proposed_limit = proposed_limit
    return {
        "proposed_limit": proposed_limit,
        "state": state
    }


# ============================================================================
# P16: Offer assembly and cross-product arbitration (§5.17, 61 decision points)
# ============================================================================

def p16_offer_assembly(
    state: DecisionState
) -> dict:
    """P16: Offer assembly (61 decision points).

    O-17: Every product's P11-P13 fan-out complete before arbitration runs.
    Returns: offers (list of valid offers), offers_assembled (count).
    Entry point 1: one product, so one offer (unless solve produces multiple tiers).
    """
    # For EP1, product 10 only: assemble one offer
    if state.affordability_pass and state.proposed_amount > 0:
        offer = {
            "product_code": state.routed_product_code,
            "amount": state.proposed_amount,
            "term_months": state.proposed_term_months,
            "monthly_instalment": state.instalment,
            "annual_percentage_rate": state.nominal_annual_rate,
            "binding_constraint": state.solve_binding_constraint
        }
        state.offers = [offer]
        state.offers_assembled = 1
    else:
        state.offers = []
        state.offers_assembled = 0

    return {
        "offers_assembled": state.offers_assembled,
        "offers": state.offers,
        "state": state
    }


# ============================================================================
# P17: Final validation (§5.18, 61 decision points)
# ============================================================================

def p17_final_validation(
    state: DecisionState
) -> dict:
    """P17: Final validation (61 decision points).

    O-18: Must be unable to read any shared intermediate (carry nothing forward).
    O-17: Assumes P16 arbitration already complete.
    EP7 (quotation): 6 of 61 assertions (no client-dependent checks).
    EP1: 14 assertions for product 10.
    Returns: validation_pass, validation_failures.
    """
    failures = []

    # Product 10 (Flex Loan) assertions for EP1 (14 total, simplified)
    if state.instalment <= 0:
        failures.append("INSTALMENT_INVALID")
    if state.proposed_amount < 2000 or state.proposed_amount > 500000:
        failures.append("AMOUNT_OUT_OF_RANGE")
    if state.proposed_term_months < 6 or state.proposed_term_months > 84:
        failures.append("TERM_OUT_OF_RANGE")
    if state.nominal_annual_rate < 0.05 or state.nominal_annual_rate > 0.30:
        failures.append("RATE_OUT_OF_RANGE")
    if state.risk_grade < 1 or state.risk_grade > 12:
        failures.append("GRADE_INVALID")

    state.validation_pass = len(failures) == 0
    state.validation_failures = failures

    return {
        "validation_pass": state.validation_pass,
        "validation_failures": failures,
        "state": state
    }


# ============================================================================
# P18: Disclosure and decision record emission (§5.19)
# ============================================================================

def p18_decision_record_emission(
    state: DecisionState,
    decision_id_override: Optional[str] = None
) -> dict:
    """P18: Disclosure and decision record emission (§5.19).

    O-20: Only after P17 passes (don't issue quotation if validation failed).
    O-21: Emission last, off critical path.
    O-19: Reason ranking after every phase that raises a reason.
    Returns: outcome_code, reason_codes, decision_id, decision record.
    """
    # Determine outcome
    if not state.validation_pass:
        state.outcome_code = 2  # Decline
        state.reason_codes = state.validation_failures
    elif state.affordability_verdict_code != 1:
        state.outcome_code = 2  # Decline
        state.reason_codes = ["UNAFFORDABLE"]
    elif state.fraud_verdict_code != 1:
        state.outcome_code = 3  # Refer or soft decline
        state.reason_codes = state.fraud_reason_codes
    elif not state.hard_eligibility_pass:
        state.outcome_code = 2  # Decline
        state.reason_codes = ["INELIGIBLE"]
    elif len(state.offers) > 0:
        state.outcome_code = 1  # Approved
        state.reason_codes = []
    else:
        state.outcome_code = 2  # Decline (no offers)
        state.reason_codes = ["NO_OFFER_GENERATED"]

    # Generate or use provided decision ID (09 §5.15 item 1)
    state.decision_id = decision_id_override or f"DECID-{datetime.now().timestamp()}"

    # Build decision record
    decision_record = {
        "decision_id": state.decision_id,
        "decision_date": state.decision_date.isoformat(),
        "entry_point_code": state.entry_point_code,
        "phase_set_id": state.phase_set_id,
        "client_id": state.client_id,
        "outcome_code": state.outcome_code,
        "reason_codes": state.reason_codes,
        "offers": state.offers,
        "routed_product": state.routed_product_code,
        "proposed_amount": state.proposed_amount,
        "proposed_term_months": state.proposed_term_months,
        "monthly_instalment": state.instalment,
        "annual_rate": state.nominal_annual_rate,
        "risk_grade": state.risk_grade,
        "affordability_verdict": state.affordability_verdict_code,
        "fraud_verdict": state.fraud_verdict_code,
        "loop_pass_count": state.loop_pass_count,
        "rate_card_version": state.rate_card_version,
    }

    return {
        "outcome_code": state.outcome_code,
        "reason_codes": state.reason_codes,
        "decision_id": state.decision_id,
        "decision_record": decision_record,
        "state": state
    }
