"""The bounded search: generate candidates in the business-authored order, evaluate them under a budget, record what happened.

The budget is a count of scenarios, never a clock, so a truncated search truncates the same way on replay. The record
says whether the search ended because it ran out of candidates or out of budget, which are different positions for a
client.
"""
from __future__ import annotations

from collections import Counter
from datetime import date

from credit_core.reason_codes import rank_codes
from decider import ParamTable, allow_fallback, missing_as, param, step

from consolidation import candidates, vocab
from consolidation.capacity import Buffers, Capacity
from consolidation.inventory import num
from consolidation.scenario import ScenarioEvaluator, remaining_term


def run_search(evaluator: ScenarioEvaluator, settleable_refs: list[int], mandatory_refs: list[int], rules: ParamTable,
               budget: int) -> tuple[list[dict], dict]:
    """Evaluate the first `budget` scenarios of the generated order; return them and the search record."""
    by_ref, policies, products = evaluator.by_ref, evaluator.policies, evaluator.products
    settleable = candidates.with_provider_facts([by_ref[r] for r in settleable_refs], evaluator.accounts)
    unavailable = {code: why for code, p in products.items() if (why := p.unavailable(evaluator.request, policies[code]))}
    max_prefix = max(p["max_accounts"] for p in policies.values())
    sets = candidates.candidate_sets(settleable, mandatory_refs, rules, max_prefix)
    order = []  # (set, rule, product, term, promotional months), the order the budget is spent in
    for refs, rule in sets:
        members = [by_ref[r] for r in refs]
        longest = max(remaining_term(a, evaluator.horizon) for a in members)
        for code in sorted(products):
            if code not in unavailable and products[code].carries(members, policies[code]):
                order += [(refs, rule, code, term, promo) for term, promo in products[code].variants(policies[code], longest)]
    evaluated = [evaluator.evaluate(i, *candidate) for i, candidate in enumerate(order[:budget], start=1)]
    rejected = Counter(code for s in evaluated for code in s["rejection_reason_codes"])
    by_severity = rank_codes(rejected, evaluator.rejection_severity, evaluator.rejection_version)[0]
    return evaluated, {
        "budget": budget, "consumption": len(evaluated), "candidate_sets": len(sets), "candidate_scenarios": len(order),
        "termination_cause": vocab.OUT_OF_BUDGET if len(order) > budget else vocab.EXHAUSTED,
        "ordering_rule_set_version": candidates.ORDERING_RULE_SET_VERSION,
        "ordering_rules": [candidates.rule(n)[0] for n in rules.rule.tolist()],
        "products_unavailable": {str(code): why for code, why in unavailable.items()},
        "rejections_by_reason": {str(code): rejected[code] for code in by_severity},
    }


def search_step(products: dict, capacity: Capacity, buffers: Buffers, rejection_severity: tuple, rejection_version: str):
    """The `search` step, pricing with `products` (product code to a product object).

    `rejection_severity` is `(code, severity rank)` for each scenario rejection code, in the `rejection_version` set.
    """

    @allow_fallback
    @step(outputs=("scenarios", "search_record"))
    def search(
        eligibility_route: str, short_circuit_applies: bool, classified_accounts: list[dict], settlements: list[dict],
        settlement_buffer_policy: dict, invariants: dict, baseline: dict, product_policies: dict, pricing_context: dict,
        settleable_account_refs: list[int], mandatory_account_refs: list[int], decision_date: date, channel_code: int,
        segment_code: float, risk_grade: float, applicant_age_years: float, requested_amount: float = missing_as(0.0),
        court_ordered_deductions: float = missing_as(0.0), consolidations_last_24_months: int = missing_as(0),
        last_balance_transfer_date: date | None = None, max_scenarios: int = param(400, ge=1, le=2_000),
        max_consolidations_24m: int = param(2, ge=1, le=3), cost_of_funds: float = param(0.08, ge=0.0),
        expected_loss_rate: float = param(0.02, ge=0.0), ordering_rules: ParamTable = candidates.ORDERING_RULES,
        overlay_stack_enabled: bool = param(True, shared_key="overlay_stack_enabled"),
    ) -> tuple[list[dict], dict]:
        """Generate and evaluate scenarios, at most `max_scenarios` of them.

        Skipped when the client is not routed to a search, or the plain request already needs no consolidation.
        """
        if eligibility_route != "proceed" or short_circuit_applies:
            cause = "not_run" if eligibility_route != "proceed" else "short_circuited"
            return [], {"termination_cause": cause, "budget": 0, "consumption": 0}
        settled_facts = {s["account_ref"]: s for s in settlements}
        request = {
            "decision_date": decision_date, "channel_code": channel_code, "segment_code": segment_code,
            "risk_grade": int(risk_grade), "applicant_age_years": applicant_age_years, "requested_amount": requested_amount,
            "court_ordered_deductions": court_ordered_deductions, "consolidations_last_24_months": consolidations_last_24_months,
            "revolving_in_arrears": sum(1 for a in classified_accounts
                                        if a["account_type_code"] in vocab.REVOLVING_TYPES and num(a, "months_in_arrears") > 0),
            "last_balance_transfer_date": last_balance_transfer_date}
        evaluator = ScenarioEvaluator(
            request, [{**a, **settled_facts.get(a["account_ref"], {})} for a in classified_accounts], invariants, baseline,
            {int(code): policy for code, policy in product_policies.items()},
            {code: product.bind(pricing_context) for code, product in products.items()}, capacity, buffers,
            settlement_buffer_policy, {"cost_of_funds": cost_of_funds, "expected_loss_rate": expected_loss_rate},
            max_consolidations_24m, overlay_stack_enabled, rejection_severity, rejection_version)
        scenarios, record = run_search(evaluator, settleable_account_refs, mandatory_account_refs, ordering_rules, max_scenarios)
        return scenarios, {**record, "overlay_stack_enabled": overlay_stack_enabled}

    return search
