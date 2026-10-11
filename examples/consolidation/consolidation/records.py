"""The shapes of the records the steps write, so the polars engine knows every column's type before the first row.

A key a record may leave out is in a `total=False` class; a key that is always there but may be null is `| None`.
"""
from __future__ import annotations

from typing import TypedDict

from consolidation import reasons


class Invariants(TypedDict):
    gross_monthly_income: float
    income_source_code: int
    income_haircut_applied: float
    statutory_deductions: float
    net_monthly_income: float
    living_expenses: float
    expense_basis_code: int
    dependants_count: float
    income_verification_tier: int
    applicant_income_evidence_gap: int
    bureau_is_stale: bool
    minimum_income_tier: int
    residual_floor_amount: float
    affordability_buffer_applied: float


class Baseline(TypedDict):
    existing_obligations: float
    discretionary_income: float
    max_affordable_instalment: float
    evidence_sufficiency_code: int
    worst_arrears_months: int
    total_instalment: float
    debt_service_ratio: float
    weighted_average_rate: float
    total_remaining_cost: float
    account_count: int
    provider_count: int
    paydown_horizon_months: int


class PlainOffer(TypedDict, total=False):
    offered_amount: float
    term_months: int
    binding_constraint_code: str
    instalment: float
    nominal_annual_rate: float
    total_cost_of_credit: float
    rate_cell_id: str


class ShortCircuitVerdicts(TypedDict):
    request_affordable_as_made: bool
    debt_service_ratio_within_ceiling: bool
    no_account_in_arrears: bool
    rate_gap_within_threshold: bool
    fewer_than_six_active_accounts: bool
    no_settleable_account_at_high_rate: bool


EligibilityGateVerdicts = TypedDict("EligibilityGateVerdicts", {
    "CON-ELIG-01": bool, "CON-ELIG-02": bool, "CON-ELIG-03": bool, "CON-ELIG-04": bool, "CON-ELIG-05": bool,
    "CON-ELIG-06": bool, "CON-ELIG-07": bool})


class SettlementComponents(TypedDict, total=False):
    capital: float
    accrued_interest: float
    fees_unpaid: float
    early_settlement_charge: float
    security_release_cost: float
    rebate: float
    quoted_amount: float


class Settlement(TypedDict):
    account_ref: int
    assumed_settlement_date: str
    quotation_reference: str | None
    quotation_expiry_date: str | None
    settlement_amount: float
    components: SettlementComponents
    amount_basis: str
    estimation_tolerance: float


class SettlementBufferPolicy(TypedDict):
    buffer_pct: float
    buffer_cap: float


class PricingContext(TypedDict):
    statutory_ceiling: float
    initiation_fee_cap: float
    monthly_service_fee_cap: float


class MarginBand(TypedDict):
    external_from: float
    base_bps: float


class FlexLoanPolicy(TypedDict):
    product_code: int
    min_accounts: int
    max_accounts: int
    min_amount: float
    max_amount: float
    min_term: int
    max_term: int
    min_relief: float
    anti_harm: float
    max_term_extension: int
    rate_ceiling: bool
    new_money_pct: float
    new_money_cap: float
    dsr_ceiling: float
    min_external: float
    min_discretionary: float
    margin_per_grade_bps: float
    margin_bands: list[MarginBand]
    candidate_terms: list[int]


class EverydayCardPolicy(TypedDict):
    product_code: int
    min_accounts: int
    max_accounts: int
    min_limit: float
    max_limit: float
    max_utilisation: float
    transfer_fee_pct: float
    minimum_payment_floor: float
    minimum_payment_pct: float
    paydown_months: int
    stress_months: int
    min_relief: float
    anti_harm: float
    rate_ceiling: bool
    new_money_pct: float
    new_money_cap: float
    dsr_ceiling: float
    min_discretionary: float
    max_revolving_in_arrears: int
    transfer_cooldown_months: int
    candidate_promotional_months: list[int]


# The keys are product codes as text; a new product is a new key here.
ProductPolicies = TypedDict("ProductPolicies", {"11": FlexLoanPolicy, "20": EverydayCardPolicy})


class BufferProbe(TypedDict):
    product_code: int
    risk_grade: int


class Intervention(TypedDict):
    intervention: int
    name: str
    status: str
    actual: float | None
    threshold: float | None
    base_threshold: float | None
    overlay_id: str | None


class Fees(TypedDict, total=False):
    initiation_fee: float
    monthly_service_fee: float
    credit_life_premium: float
    margin_bps: float
    transfer_fee: float
    reversion_rate: float
    reversion_cell_id: str
    promotional_months: int


class Position(TypedDict):
    monthly_instalment: float
    weighted_average_rate: float
    longest_remaining_term_months: float
    total_cost: float
    accounts: int
    providers: int


class Measures(TypedDict):
    new_money_ratio: float
    instalment_relief_ratio: float
    total_cost_increase_ratio: float
    bank_value_ratio: float
    accounts_exited_proportion: float
    weighted_rate_reduction: float
    providers_exited_proportion: float


class Scenario(TypedDict, total=False):
    scenario_id: int
    settlement_set: list[int]
    generating_rule: str
    product_code: int
    term_months: int
    promotional_months: int
    settlement_total: float
    settlement_buffer: float
    new_money: float
    external_proportion: float
    rejection_reason_codes: list[int]
    primary_rejection_reason_code: int | None
    interventions: list[Intervention]
    viable: bool
    offered_amount: float
    nominal_annual_rate: float
    instalment: float
    tested_payment: float
    total_cost_of_credit: float
    effective_annual_rate: float
    rate_cell_id: str
    rate_card_version: str
    fees: Fees
    existing_obligations_after: float
    discretionary_income: float
    max_affordable_instalment: float
    affordability_verdict_code: int
    instalment_relief: float
    total_cost_delta: float
    settled_longest_remaining_term_months: float
    bank_expected_value: float
    after: Position
    measures: Measures
    unpriced_reason: str


class ChosenScenario(Scenario, total=False):
    objective_score: float


class ChosenOffer(ChosenScenario, total=False):
    binding_constraint_code: str
    settlement_details: list[Settlement]


# The scenario rejection codes the code can raise: a failed intervention is 1000 + its number.
_REJECTIONS = [str(1000 + n) for n in (1, 3, 4, 5, 6, 8, 9, 10, 13, 14)] + [
    str(c) for c in (reasons.NOT_PRICED, reasons.AFFORDABILITY_FAIL, reasons.OUTSIDE_PRODUCT_LIMITS)]
RejectionsByReason = TypedDict("RejectionsByReason", {code: int for code in _REJECTIONS}, total=False)
ProductsUnavailable = TypedDict("ProductsUnavailable", {"11": str, "20": str}, total=False)


class SearchRecord(TypedDict, total=False):
    budget: int
    consumption: int
    candidate_sets: int
    candidate_scenarios: int
    termination_cause: str
    ordering_rule_set_version: str
    ordering_rules: list[str]
    products_unavailable: ProductsUnavailable
    rejections_by_reason: RejectionsByReason
    overlay_stack_enabled: bool


class ObjectiveValues(TypedDict):
    new_money: float
    min_commitment: float
    min_total_cost: float
    bank_value: float
    client_outcome: float


class ScoredScenario(TypedDict):
    scenario_id: int
    objective_score: float
    components: ObjectiveValues


class Versus(TypedDict):
    winner: int
    runner_up: int
    score_difference: float
    weighted_component_differences: ObjectiveValues
    decisive_component: str


class ShadowBest(TypedDict, total=False):
    scenario_id: int
    client_outcome_score: float


class BeforeAndAfter(TypedDict, total=False):
    before: Position
    after: Position
    total_cost_change: float
    total_cost_change_pct: float
    total_cost_note: str
