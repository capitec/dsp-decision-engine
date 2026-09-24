"""
Core consolidation logic: settleability, settlement amounts, scenario generation and evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional, Dict, List, Tuple, Set
from enum import IntEnum
import itertools
import heapq
from collections import defaultdict

# Settleability codes (§5.2)
class SettleabilityCode(IntEnum):
    INTERNAL = 0  # Bank's own account
    QUOTATION_HELD = 1  # External, unexpired quotation
    QUOTATION_OBTAINABLE = 2  # External, can be quoted
    SECURITY_RELEASE = 3  # Secured, needs security release
    PARTIALLY_SETTLEABLE = 8  # Revolving
    BLOCKED_BY_PROVIDER = 4  # Provider doesn't settle
    BLOCKED_BY_POLICY = 5  # New account, etc
    BLOCKED_BY_STATUS = 6  # Disputed, under debt review
    UNKNOWN = 7  # Provider unknown or data missing


# Assessment modes (§4.7)
class AssessmentMode(IntEnum):
    CONSOLIDATION_ON_REQUEST = 1
    CONSOLIDATION_OFFERED = 2
    RESTRUCTURE = 3
    BATCH_IDENTIFICATION = 4


# Rejection reason codes (§5.7)
class RejectionReasonCode(IntEnum):
    # CON-INT violations
    FEWER_THAN_2_ACCOUNTS = 3001
    EXTERNAL_PROPORTION_BELOW_60 = 3002
    ADVANCE_BELOW_MIN = 3003
    ADVANCE_ABOVE_MAX = 3004
    INSTALMENT_RELIEF_BELOW_FLOOR = 3005
    MAX_ACCOUNTS_EXCEEDED = 3010
    NEWLY_OPENED_ACCOUNT = 3011
    ANTI_HARM_BREACH = 3012
    TERM_EXTENSION_EXCEEDS_LIMIT = 3013
    RATE_EXCEEDS_CEILING = 3014
    DSR_EXCEEDS_CEILING = 3015
    DISCRETIONARY_INCOME_BELOW_FLOOR = 3016
    NON_REVOLVING_ACCOUNT_IN_SET = 3020  # Product 20
    TRANSFERRED_BALANCE_EXCEEDS_LIMIT = 3021
    MULTIPLE_REVOLVING_IN_ARREARS = 3022
    STRESSED_PAYMENT_UNAFFORDABLE = 3023


# Product codes
PRODUCT_FLEX_CONSOLIDATION = 11
PRODUCT_BALANCE_TRANSFER = 20
PRODUCTS_AVAILABLE = [PRODUCT_FLEX_CONSOLIDATION, PRODUCT_BALANCE_TRANSFER]


@dataclass
class Account:
    """Individual obligation in the inventory."""
    account_ref: int
    account_type_code: int
    provider_code: int
    is_internal: bool
    balance: Optional[float]
    instalment: Optional[float]
    nominal_annual_rate: Optional[float]
    remaining_term_months: Optional[int]
    months_in_arrears: int
    opened_date: Optional[date]
    is_secured: bool
    account_status_code: int
    is_disputed: bool

    # Derived
    settleability_code: Optional[SettleabilityCode] = None
    settlement_amount: Optional[float] = None
    quotation_reference: Optional[str] = None
    quotation_expiry_date: Optional[date] = None
    is_client_mandatory: bool = False
    is_client_excluded: bool = False


@dataclass
class Scenario:
    """A settlement set + product + term combination."""
    scenario_id: int
    settlement_set: List[int]  # Account refs
    product_code: int
    term_months: int
    new_money: float

    # Pricing
    offered_amount: Optional[float] = None
    nominal_annual_rate: Optional[float] = None
    instalment: Optional[float] = None
    total_cost_of_credit: Optional[float] = None

    # Affordability
    new_existing_obligations: Optional[float] = None
    new_discretionary_income: Optional[float] = None
    affordability_verdict_code: Optional[int] = None

    # Measures
    instalment_relief: Optional[float] = None  # Monthly savings
    total_cost_delta: Optional[float] = None  # Additional total cost
    new_weighted_avg_rate: Optional[float] = None
    new_debt_service_ratio: Optional[float] = None
    bank_expected_value: Optional[float] = None
    client_outcome_score: Optional[float] = None

    # Status
    is_viable: bool = False
    rejection_reason_codes: List[int] = field(default_factory=list)

    # Tie-breaking
    _scenario_key: Tuple = field(default_factory=tuple)

    def __lt__(self, other):
        """For heap: higher score better (negate for min-heap behavior)."""
        return self._scenario_key > other._scenario_key


@dataclass
class AssessmentResult:
    """Output of the consolidation assessment."""
    decision_id: str
    client_id: int
    decision_date: date
    is_eligible: bool
    decline_reason_codes: List[int] = field(default_factory=list)

    # Settlement analysis
    accounts: List[Account] = field(default_factory=list)
    settleable_count: int = 0
    baseline_instalment: float = 0.0
    baseline_weighted_rate: float = 0.0
    baseline_total_cost: float = 0.0

    # Search
    scenarios_evaluated: int = 0
    scenarios_budget: int = 0
    budget_exhausted: bool = False

    # Results
    scenarios_all: List[Scenario] = field(default_factory=list)
    scenarios_top_3: List[Scenario] = field(default_factory=list)
    winner: Optional[Scenario] = None
    objective_id: int = 2

    # Evidence
    evidence: Dict = field(default_factory=dict)


def classify_settleability(
    account: Account,
    decision_date: date,
    months_since_open_threshold: int = 3
) -> SettleabilityCode:
    """Classify account settleability per §5.2."""

    # Order matters: first match wins

    # 7: Unknown
    if account.provider_code is None:
        return SettleabilityCode.UNKNOWN

    # 6: Blocked by status
    if account.is_disputed or account.account_status_code in (3, 4, 5, 6, 7):
        # 3=in dispute, 4=handed over, 5=debt review, 6=legal, 7=written off
        return SettleabilityCode.BLOCKED_BY_STATUS

    # 5: Blocked by policy
    if account.opened_date:
        months_open = (decision_date - account.opened_date).days / 30
        if months_open < months_since_open_threshold:
            return SettleabilityCode.BLOCKED_BY_POLICY

    # 4: Blocked by provider (stub: no provider has this in demo)
    if account.provider_code > 200:
        return SettleabilityCode.BLOCKED_BY_PROVIDER

    # 3: Secured
    if account.is_secured:
        return SettleabilityCode.SECURITY_RELEASE

    # 8: Partially settleable (revolving)
    if account.account_type_code in (20, 21):  # Revolving/credit card
        return SettleabilityCode.PARTIALLY_SETTLEABLE

    # 0: Internal
    if account.is_internal:
        return SettleabilityCode.INTERNAL

    # 1/2: External (stub: assume obtainable)
    if account.quotation_expiry_date and account.quotation_expiry_date >= decision_date:
        return SettleabilityCode.QUOTATION_HELD

    return SettleabilityCode.QUOTATION_OBTAINABLE


def derive_settlement_amount(account: Account, decision_date: date) -> float:
    """
    Derive settlement amount per §5.3.

    In full implementation: balance + accrued interest + fees + early settlement charge
    - rebates - security release cost.

    For demo: use balance + 1.5% buffer capped R2500, plus settlement charge.
    """
    if account.balance is None or account.balance <= 0:
        return 0.0

    # Base: balance
    amount = account.balance

    # Settlement buffer: 1.5%, capped R2500
    buffer = min(amount * 0.015, 2500.0)
    amount += buffer

    # Early settlement charge (stub): assume 2% for high-rate accounts
    if account.nominal_annual_rate and account.nominal_annual_rate > 24:
        charge = amount * 0.02
        amount += charge

    return amount


def baseline_assessment(
    accounts: List[Account],
    gross_income: float,
    net_income: float,
    existing_obligations: float,
    decision_date: date,
    requested_amount: Optional[float] = None
) -> Dict:
    """
    Baseline (do-nothing) position per §5.4.

    Returns baseline measures and short-circuit evaluation.
    """

    # Current state
    current_instalment = sum(a.instalment for a in accounts if a.instalment and not a.is_disputed) or 0.0

    # Weighted average rate
    total_balance = sum(a.balance for a in accounts if a.balance) or 1.0
    avg_rate = sum(
        (a.balance or 0) * (a.nominal_annual_rate or 12)
        for a in accounts if a.balance
    ) / total_balance if total_balance > 0 else 12.0

    # Total remaining cost (simplified: remaining instalments × term for term loans)
    total_remaining_cost = sum(
        (a.instalment or 0) * (a.remaining_term_months or 36)
        for a in accounts if a.remaining_term_months
    ) or 0.0

    # Longest term
    longest_term = max([a.remaining_term_months for a in accounts if a.remaining_term_months], default=0)

    # DSR
    dsr = existing_obligations / net_income if net_income > 0 else 1.0

    # In arrears
    in_arrears = any(a.months_in_arrears > 0 for a in accounts)

    # Short-circuit conditions (§5.4)
    # If all pass: consolidation not needed
    basic_affordable = requested_amount is not None  # Stub: would check vs max_affordable_instalment
    post_dsr_ok = (existing_obligations + (100 if requested_amount else 0)) / net_income <= 0.40
    no_arrears = not in_arrears
    rate_not_too_bad = avg_rate - 10 <= 300  # New rate would be ~10%, so delta <= 300bps
    few_accounts = len(accounts) < 6
    no_high_rate = not any((a.nominal_annual_rate or 0) >= 28 for a in accounts)

    short_circuit = (
        basic_affordable and post_dsr_ok and no_arrears and
        rate_not_too_bad and few_accounts and no_high_rate
    )

    return {
        "current_instalment": current_instalment,
        "weighted_average_rate": avg_rate,
        "longest_remaining_term": longest_term,
        "total_remaining_cost": total_remaining_cost,
        "debt_service_ratio": dsr,
        "in_arrears": in_arrears,
        "short_circuit_to_granting": short_circuit,
    }


def generate_candidate_scenarios(
    accounts: List[Account],
    settleable_accounts: List[Account],
    budget: int = 400,
    decision_date: date = None,
    con_int_01_max: int = 8,
) -> List[Scenario]:
    """
    Generate candidate scenarios per §5.5 using H1-H8 heuristics.

    Returns scenarios in evaluation order within budget.
    """

    candidates = []
    scenario_id = 0

    # Mandatory accounts (client insisted)
    mandatory = [a.account_ref for a in settleable_accounts if a.is_client_mandatory]

    # Ordering heuristics (simplified)
    # H1: Highest effective rate first
    # H3: Shortest remaining term last (exclude)

    # Generate settlement sets
    # Start with empty set (baseline)
    candidates.append(Scenario(
        scenario_id=scenario_id,
        settlement_set=[],
        product_code=PRODUCT_FLEX_CONSOLIDATION,
        term_months=60,
        new_money=0.0,
    ))
    scenario_id += 1

    # Sort by H1: highest rate
    sorted_by_rate = sorted(
        settleable_accounts,
        key=lambda a: -(a.nominal_annual_rate or 0)
    )

    # Prefixes of H1 ordering
    for k in range(1, min(len(sorted_by_rate) + 1, con_int_01_max + 1)):
        settlement_set = [a.account_ref for a in sorted_by_rate[:k]]

        # For each set, try different products and terms
        for product_code in PRODUCTS_AVAILABLE:
            for term in [24, 36, 48, 60, 72]:
                candidates.append(Scenario(
                    scenario_id=scenario_id,
                    settlement_set=settlement_set,
                    product_code=product_code,
                    term_months=term,
                    new_money=0.0,
                ))
                scenario_id += 1

                if len(candidates) >= budget * 2:  # Generate more than needed
                    break
            if len(candidates) >= budget * 2:
                break
        if len(candidates) >= budget * 2:
            break

    # Add client nominated set
    if mandatory:
        for product_code in PRODUCTS_AVAILABLE:
            for term in [36, 60, 72]:
                candidates.append(Scenario(
                    scenario_id=scenario_id,
                    settlement_set=mandatory,
                    product_code=product_code,
                    term_months=term,
                    new_money=0.0,
                ))
                scenario_id += 1

    return candidates[:budget]


def evaluate_scenario(
    scenario: Scenario,
    accounts: List[Account],
    setteable_accounts: List[Account],
    decision_date: date,
    net_income: float,
    living_expenses: float,
    thresholds: Dict,
) -> Scenario:
    """
    Evaluate a single scenario per §5.6.

    Returns scenario with viability verdict and rejection reasons.
    """

    # Get settled accounts
    settled = [a for a in accounts if a.account_ref in scenario.settlement_set]
    retained = [a for a in accounts if a.account_ref not in scenario.settlement_set]

    # Compute settlement amount
    settlement_total = sum(
        (a.settlement_amount or 0) for a in settled
    )

    if scenario.settlement_set:
        # Product routing checks (simplified)
        if scenario.product_code == PRODUCT_FLEX_CONSOLIDATION:
            # At least 2 accounts
            if len(scenario.settlement_set) < 2:
                scenario.rejection_reason_codes.append(RejectionReasonCode.FEWER_THAN_2_ACCOUNTS)
                return scenario

            # At least 60% to external creditors
            external_amount = sum(
                a.settlement_amount for a in settled if not a.is_internal
            ) or 0.0
            if settlement_total > 0 and external_amount / settlement_total < 0.60:
                scenario.rejection_reason_codes.append(RejectionReasonCode.EXTERNAL_PROPORTION_BELOW_60)

            # Max 8 accounts
            if len(scenario.settlement_set) > thresholds.get("con_int_01_max_accounts", 8):
                scenario.rejection_reason_codes.append(RejectionReasonCode.MAX_ACCOUNTS_EXCEEDED)

        elif scenario.product_code == PRODUCT_BALANCE_TRANSFER:
            # All revolving
            if any(a.account_type_code not in (20, 21) for a in settled):
                scenario.rejection_reason_codes.append(RejectionReasonCode.NON_REVOLVING_ACCOUNT_IN_SET)

    # If rejected, mark not viable
    if scenario.rejection_reason_codes:
        scenario.is_viable = False
        return scenario

    # Compute new obligations
    new_instalment = 300.0  # Stub: would call product pricing logic
    new_total_cost = 20000.0  # Stub

    # Affordability
    new_obligations = sum(a.instalment for a in retained if a.instalment) or 0.0
    new_obligations += new_instalment

    new_discretionary = net_income - living_expenses - new_obligations

    # Instalment relief
    old_instalment = sum(a.instalment for a in settled if a.instalment) or 0.0
    scenario.instalment_relief = old_instalment - new_instalment

    # Total cost delta
    old_remaining_cost = sum(
        (a.instalment or 0) * (a.remaining_term_months or 36)
        for a in settled
    ) or 0.0
    scenario.total_cost_delta = new_total_cost - old_remaining_cost

    # Mark viable if basic checks pass
    scenario.is_viable = True
    scenario.new_existing_obligations = new_obligations
    scenario.new_discretionary_income = new_discretionary
    scenario.instalment = new_instalment
    scenario.total_cost_of_credit = new_total_cost

    return scenario


def select_winner(scenarios: List[Scenario], objective_id: int) -> Scenario:
    """
    Select winning scenario per §5.8.

    Returns top viable scenario under the objective.
    """

    viable = [s for s in scenarios if s.is_viable]
    if not viable:
        return None

    # Simplified objective: OBJ-02 (minimize instalment)
    if objective_id == 2:
        return min(viable, key=lambda s: (s.instalment or 9999, s.total_cost_of_credit or 9999))

    return viable[0] if viable else None


def assess_consolidation(request: Dict) -> Dict:
    """Main consolidation assessment logic."""

    import uuid
    decision_id = str(uuid.uuid4())
    client_id = request["client_id"]
    decision_date = request["decision_date"]

    # Parse decision_date if it's a string
    if isinstance(decision_date, str):
        decision_date = date.fromisoformat(decision_date)

    result = AssessmentResult(
        decision_id=decision_id,
        client_id=client_id,
        decision_date=decision_date,
        is_eligible=True,
    )

    # Parse accounts
    accounts = []
    for acc_data in request.get("accounts", []):
        # Parse opened_date if it's a string
        opened_date = acc_data.get("opened_date", decision_date - timedelta(days=180))
        if isinstance(opened_date, str):
            opened_date = date.fromisoformat(opened_date)

        acc = Account(
            account_ref=acc_data.get("account_ref", 1),
            account_type_code=acc_data.get("account_type_code", 10),
            provider_code=acc_data.get("provider_code", 1),
            is_internal=acc_data.get("is_internal", False),
            balance=acc_data.get("balance", 10000.0),
            instalment=acc_data.get("instalment", 500.0),
            nominal_annual_rate=acc_data.get("nominal_annual_rate", 15.0),
            remaining_term_months=acc_data.get("remaining_term_months", 24),
            months_in_arrears=acc_data.get("months_in_arrears", 0),
            opened_date=opened_date,
            is_secured=acc_data.get("is_secured", False),
            account_status_code=acc_data.get("account_status_code", 1),
            is_disputed=acc_data.get("is_disputed", False),
        )
        accounts.append(acc)

    result.accounts = accounts

    # Classify settleability
    for acc in accounts:
        acc.settleability_code = classify_settleability(acc, decision_date)
        acc.settlement_amount = derive_settlement_amount(acc, decision_date)

        # Apply client nominations
        if acc.account_ref in request.get("client_nominated_settle", []):
            acc.is_client_mandatory = True
        if acc.account_ref in request.get("client_excluded_settle", []):
            acc.is_client_excluded = True

    settleable = [a for a in accounts if a.settleability_code in (0, 1, 2, 3, 8)]
    result.settleable_count = len(settleable)

    # Baseline
    baseline = baseline_assessment(
        accounts,
        request.get("gross_monthly_income", 15000.0),
        request.get("net_monthly_income", 11000.0),
        request.get("existing_obligations", 2500.0),
        decision_date,
        request.get("requested_amount"),
    )

    result.baseline_instalment = baseline["current_instalment"]
    result.baseline_weighted_rate = baseline["weighted_average_rate"]
    result.baseline_total_cost = baseline["total_remaining_cost"]

    # Candidate generation
    budget = request.get("scenario_budget", 400)
    candidates = generate_candidate_scenarios(
        accounts,
        settleable,
        budget,
        decision_date,
        request.get("policy_thresholds", {}).get("con_int_01_max_accounts", 8),
    )

    # Evaluate scenarios
    result.scenarios_budget = budget
    result.scenarios_evaluated = 0

    for scenario in candidates[:budget]:
        scenario = evaluate_scenario(
            scenario,
            accounts,
            settleable,
            decision_date,
            request.get("net_monthly_income", 11000.0),
            request.get("living_expenses", 5000.0),
            request.get("policy_thresholds", {}),
        )
        result.scenarios_all.append(scenario)
        result.scenarios_evaluated += 1

    result.budget_exhausted = len(candidates) > budget

    # Select winner
    objective_id = request.get("objective_id", 2)
    winner = select_winner(result.scenarios_all, objective_id)
    result.winner = winner
    result.objective_id = objective_id

    # Top 3
    viable = [s for s in result.scenarios_all if s.is_viable]
    result.scenarios_top_3 = viable[:3]

    # Build response
    return {
        "decision_id": decision_id,
        "client_id": client_id,
        "decision_date": decision_date.isoformat(),
        "is_eligible": result.is_eligible,
        "settleable_count": result.settleable_count,
        "baseline_instalment": result.baseline_instalment,
        "baseline_weighted_rate": result.baseline_weighted_rate,
        "scenarios_evaluated": result.scenarios_evaluated,
        "scenarios_budget": result.scenarios_budget,
        "budget_exhausted": result.budget_exhausted,
        "winner": {
            "scenario_id": winner.scenario_id if winner else None,
            "settlement_set": winner.settlement_set if winner else [],
            "product_code": winner.product_code if winner else None,
            "term_months": winner.term_months if winner else None,
            "instalment_relief": winner.instalment_relief if winner else None,
            "total_cost_delta": winner.total_cost_delta if winner else None,
        } if winner else None,
        "top_3_count": len(result.scenarios_top_3),
        "objective_id": result.objective_id,
    }
