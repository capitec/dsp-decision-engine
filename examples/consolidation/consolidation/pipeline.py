"""A consolidation assessment: which of a client's debts to settle, under which product, and why.

`decider build` and `decider serve` use `build`, which receives the product 11 rate card, the two Everyday Card rate
documents and the settleability rules from `configs/<version>/`.
"""
from __future__ import annotations

from assessment import modes
from assessment.pipeline import evidence_unit
from credit_core import fees
from credit_core.rate_card import generate_flex_loan_card
from decider import flow, param
from loan_granting.pricing import CreditLifeIndex, RateCardIndex

from consolidation import (
    baseline,
    eligibility,
    inventory,
    objective,
    outcome,
    products,
    search,
    settlement,
)
from consolidation.capacity import Buffers, Capacity
from consolidation.documents import shipped
from consolidation.everyday_card import CardRates, EverydayCard, everyday_card_policy
from consolidation.flex_loan import FlexLoanConsolidation, flex_loan_policy
from consolidation.vocab import PRODUCT_EVERYDAY_CARD, PRODUCT_FLEX_LOAN_CONSOLIDATION

INVENTORY = ("classified_accounts", "settleable_account_refs", "mandatory_account_refs", "excluded_account_refs",
             "nomination_conflicts", "settleable_count", "unknown_count", "settleable_obligation_proportion", "settlements",
             "settlement_buffer_policy", "eligibility_gate_verdicts", "eligibility_route")
BASELINE = ("invariants", "baseline", "plain_offer", "short_circuit_verdicts", "short_circuit_applies")
SEARCH = ("product_policies", "pricing_context", "scenarios", "search_record", "objective_weights", "objective_overlay_ids",
          "scored_scenarios", "runners_up", "indifference_band_triggered", "winner_vs_runners_up", "shadow_best_scenario")
DECISION = ("outcome_code", "decline_reason_codes", "primary_reason_code", "reason_registry_version", "chosen_offer",
            "conditions", "offer_valid_until", "before_and_after", "execution_package")
EVIDENCE = ("fee_caps_cell_id", "norm_table_version", "statutory_norm_cell_id", "internal_norm_cell_id",
            "applicant1_tax_table_cell_id", "buffer_grid_cell_id", "residual_floor_cell_id")


def assessment_mode_code(mode: int = param(modes.SCENARIO, ge=1, le=4)) -> int:
    return mode


def product_code(evidence_product: int = param(PRODUCT_FLEX_LOAN_CONSOLIDATION)) -> int:
    """The product whose affordability buffer the shared evidence is derived under; each scenario then reads its own."""
    return evidence_product


def statutory_ceiling(repo_rate: float = param(0.07, ge=0.0, le=0.30), margin_over_repo: float = param(0.21, ge=0.0)) -> float:
    """The highest nominal rate the law allows: a margin over the repo rate. Compliance moves it on gazette."""
    return round(repo_rate + margin_over_repo, 4)


def build(rate_card_product11, card_promotional_rates, card_reversion_rates, settleability_rules, client_reason_codes=None,
          scenario_rejection_codes=None):
    """Each argument is the config document of that name, loaded by `decider`. The two reason code sets default to the
    documents shipped in `configs/0.1.0/`."""
    client_reasons = shipped("client_reason_codes") if client_reason_codes is None else client_reason_codes
    rejections = shipped("scenario_rejection_codes") if scenario_rejection_codes is None else scenario_rejection_codes
    capacity, buffers = Capacity(), Buffers()
    consolidation_card = RateCardIndex.from_card(rate_card_product11)
    # ponytail: the Flex Loan card is regenerated, not read from a config document; load it from one when Treasury refreshes it
    plain_card = RateCardIndex(generate_flex_loan_card("rc-2026.09")["rows"])
    carriers = {PRODUCT_FLEX_LOAN_CONSOLIDATION: FlexLoanConsolidation(consolidation_card, CreditLifeIndex()),
                PRODUCT_EVERYDAY_CARD: EverydayCard(CardRates(card_promotional_rates, card_reversion_rates))}
    return flow(
        inventory.classify_unit(settleability_rules), settlement.settlements,
        assessment_mode_code, product_code, evidence_unit(), baseline.invariants,
        statutory_ceiling, fees.fee_caps_version_step, fees.build_fee_caps_table(), products.pricing_context,
        baseline.baseline_step(capacity), baseline.plain_granting_step(plain_card, consolidation_card),
        eligibility.eligibility_gates, flex_loan_policy, everyday_card_policy, products.product_policies,
        search.search_step(carriers, capacity, buffers, tuple((c.code, c.severity_rank) for c in rejections.codes), rejections.version),
        objective.selection, outcome.outcome, client_reasons,
        name="consolidation",
    ).emit(*INVENTORY, *BASELINE, *SEARCH, *DECISION, *EVIDENCE)
