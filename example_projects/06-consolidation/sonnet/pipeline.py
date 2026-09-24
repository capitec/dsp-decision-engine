"""Consolidation and restructure (spec 06). SCOPE.md slice: obligation inventory and
settleability (§5.2-5.3), baseline assessment with the short-circuit to project 03
(§5.4), candidate scenario generation under budget (§5.5), per-scenario evaluation
re-deriving obligations and calling project 02 in scenario mode (§5.6), the
configurable objective with one re-weight overlay (§5.8), for product 11 (Flex Loan
Consolidation) and product 20 (Everyday Card balance transfer) only.

Five stages, wired as one `dag` (`decider` resolves the real dependency order from
reads/writes, not list order -- 00/02/03's own precedent):

    eligibility_gates (pre-search gates) -> settleability (§5.2) ->
    settlement_derivation (§5.3) -> orchestration (§5.4-§5.8, one big frame_step,
    see orchestration.py's own docstring for why) -> final_outcome_code (combines
    eligibility routing with the search's own outcome)
"""
from __future__ import annotations

from decider import dag

from consolidation import eligibility, inventory, outcome, settlement
from consolidation.orchestration import build_orchestration_step


def build(rate_card_flex_loan, rate_card_product11=None):
    """`rate_card_flex_loan` is project 00's original Flex Loan (product 10) card --
    needed only for the short-circuit hand-off to project 03's pipeline (`build()`
    signature there takes the same argument). `rate_card_product11` is this
    project's own card (defaults to an in-process regeneration if not supplied,
    e.g. in a quick test). Product 20's two small tables are built in-process
    unconditionally, following project 00's own precedent of externalising only
    the one dominant pricing table to `configs/`."""
    product11_rows = rate_card_product11.rows.df.to_dicts() if rate_card_product11 is not None else None
    return dag(
        inventory.settleability,
        settlement.settlement_derivation,
        eligibility.eligibility_gates,
        build_orchestration_step(rate_card_flex_loan, product11_rows),
        outcome.final_outcome_code_step,
        name="consolidation",
    ).emit(
        "decision_id", "eligibility_route", "eligibility_gate_ids", "eligibility_gate_verdicts",
        "settleability_account_refs", "settleability_codes", "settleability_rule_ids",
        "settleable_account_refs", "settleable_count", "unknown_count",
        "settlement_account_refs", "settlement_amounts", "settlement_total", "settlement_buffer_amount",
        "existing_obligations_baseline", "baseline_total_instalment", "baseline_weighted_average_rate",
        "baseline_total_remaining_cost", "baseline_account_count", "baseline_provider_count",
        "achievable_consolidation_rate", "short_circuit_applies", "short_circuit_condition_names",
        "short_circuit_condition_verdicts", "post_advance_debt_service_ratio",
        "net_monthly_income", "living_expenses", "evidence_sufficiency_code",
        "plain_grant_via_project03",
        "search_budget", "search_consumption", "search_termination_cause", "ordering_rule_set_version",
        "objective_id", "objective_weight_names", "objective_weight_values", "objective_overlay_ids",
        "indifference_band_triggered",
        "shadow_best_scenario_id", "shadow_best_objective_score",
        "scenario_ids", "scenario_settlement_sets", "scenario_generating_rules", "scenario_product_codes",
        "scenario_term_months", "scenario_viable", "scenario_rejection_reason_codes",
        "scenario_offered_amounts", "scenario_nominal_rates", "scenario_instalments",
        "scenario_total_costs_of_credit", "scenario_objective_scores",
        "runner_up_scenario_ids", "runner_up_objective_scores", "runner_up_product_codes",
        "chosen_scenario_id", "chosen_settlement_set", "chosen_product_code", "chosen_term_months",
        "chosen_offered_amount", "chosen_nominal_annual_rate", "chosen_instalment",
        "chosen_total_cost_of_credit", "chosen_instalment_relief", "chosen_total_cost_delta",
        "chosen_objective_score",
        "outcome_code", "primary_reason_code", "decline_reason_codes", "reason_registry_version",
    )
