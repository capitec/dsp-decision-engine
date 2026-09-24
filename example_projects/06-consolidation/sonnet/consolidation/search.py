"""Stage 5.5 -- candidate scenario generation (spec 06 §5.5).

A **bounded search**: 18 settleable accounts is 262 143 non-empty settlement
sets, times up to 4 products, times up to 9 terms -- 9.4 million possible
scenarios, of which at most 400 are ever evaluated (§8: the interactive
budget). Exhaustive search is not attempted; instead, eight business-authored
ordering rules (H1-H8) each contribute a *prefix sequence* of candidate sets,
plus the empty set, the client's nominated set, and the full settleable set
(§5.5 "What the search must generate") -- deterministic, because the same
client, inputs, `decision_date` and overlay stack must produce the same
candidate list in the same order every time (requirement 3).

Candidate *sets* are generated first, independent of product; product
routing (§5.6.3) and term enumeration happen per set, so the same ordering
logic serves every product without being duplicated per product.
"""
from __future__ import annotations

from dataclasses import dataclass

from consolidation import vocab

ORDERING_RULE_SET_VERSION = "ordering-rules-2026.09"

CARD_TYPES = frozenset({10, 11})  # revolving/card account types (product 20 eligible)

# Bounded set of candidate terms product 11 considers per settlement set (§5.5:
# "the product minimum, the maximum permitted by CON-INT-05, and the terms that
# bracket the affordability constraint" -- a small declared ladder, not a scan).
PRODUCT11_CANDIDATE_TERMS = (12, 24, 36, 48, 60, 72, 84)

_MAX_PREFIX_K = 8  # matches CON-INT-01's default max-accounts-settled ceiling


@dataclass(frozen=True)
class SettleableAccount:
    account_ref: int
    settlement_amount: float
    instalment: float
    nominal_annual_rate: float
    remaining_term_months: float | None
    provider_code: int
    is_internal: bool
    months_in_arrears: int
    account_type_code: int
    is_mandatory: bool

    @property
    def is_card_type(self) -> bool:
        return self.account_type_code in CARD_TYPES


def _relief_per_rand(a: SettleableAccount) -> float:
    return (a.instalment / a.settlement_amount) if a.settlement_amount else 0.0


# H1-H8 (§5.5), each a sort key producing a full ordering of settleable accounts;
# the *prefixes* of that ordering (top 1, top 2, ..., top k) are the candidates.
ORDERING_RULES = {
    "H1_highest_rate_first": lambda a: -a.nominal_annual_rate,
    "H2_highest_relief_per_rand": lambda a: -_relief_per_rand(a),
    "H3_shortest_term_last": lambda a: -(a.remaining_term_months or 0),  # short terms sort to the end
    "H5_provider_in_arrears_first": lambda a: (0 if a.months_in_arrears > 0 else 1, -a.nominal_annual_rate),
    "H6_quotation_held_first": lambda a: -a.nominal_annual_rate,  # settleability already narrows to held/quotable
    "H7_full_provider_exit_first": lambda a: -a.settlement_amount,  # larger balances -> larger relationships
    "H8_highest_reaccumulation_risk": lambda a: (0 if a.is_card_type else 1, -a.nominal_annual_rate),
}


def generate_candidate_sets(
    accounts: list[SettleableAccount], mandatory_refs: list[int],
) -> list[tuple[frozenset, str]]:
    """Every candidate settlement set, in generation order, paired with the rule that
    produced it (for the recorded trace, §5.5 requirement 5). Deduplicated: a set
    produced by two rules is recorded once, at its first (highest-priority) occurrence."""
    mandatory = frozenset(mandatory_refs)
    seen: set[frozenset] = set()
    candidates: list[tuple[frozenset, str]] = []

    def add(s: frozenset, rule: str) -> None:
        if s in seen:
            return
        if mandatory and not mandatory.issubset(s):
            return  # "a mandatory account exists: only sets that contain it" (§5.5)
        seen.add(s)
        candidates.append((s, rule))

    add(frozenset(), "empty_set")  # always first: the baseline comparator

    if mandatory:
        add(mandatory, "client_nominated")

    max_k = min(len(accounts), _MAX_PREFIX_K)
    for rule_name, key in ORDERING_RULES.items():
        ordered = sorted(accounts, key=key)
        for k in range(1, max_k + 1):
            prefix = frozenset(a.account_ref for a in ordered[:k]) | mandatory
            add(prefix, rule_name)

    add(frozenset(a.account_ref for a in accounts), "full_settleable_set")

    return candidates


def route_products(accounts_in_set: list[SettleableAccount]) -> list[int]:
    """§5.6.3: which products *can* carry this settlement set. Only 11 and 20 are
    priced this slice (SCOPE.md); 30 and 40 are recognised as routable in principle
    but never evaluated (no collateral/bond data collected this slice)."""
    products = []
    if len(accounts_in_set) >= 1:
        products.append(vocab.PRODUCT_FLEX_CONSOLIDATION)
    if accounts_in_set and all(a.is_card_type for a in accounts_in_set):
        products.append(vocab.PRODUCT_BALANCE_TRANSFER)
    return products


def candidate_terms(product_code: int, longest_settled_remaining_term: float, max_term_extension: int) -> list[int]:
    if product_code == vocab.PRODUCT_FLEX_CONSOLIDATION:
        ceiling = min(84, (longest_settled_remaining_term or 0) + max_term_extension)
        return sorted({t for t in PRODUCT11_CANDIDATE_TERMS if 12 <= t <= max(ceiling, 12)}) or [12]
    if product_code == vocab.PRODUCT_BALANCE_TRANSFER:
        from consolidation.rate_cards import PRODUCT20_PROMO_DURATIONS
        return list(PRODUCT20_PROMO_DURATIONS)
    return []


def generate_scenarios(
    accounts: list[SettleableAccount], mandatory_refs: list[int], max_term_extension: int, budget: int,
) -> tuple[list[dict], str]:
    """The full (settlement_set, product, term) candidate list, in the total order the
    budget is consumed against, and the termination cause (§5.5 requirement 2)."""
    by_ref = {a.account_ref: a for a in accounts}
    candidate_sets = generate_candidate_sets(accounts, mandatory_refs)

    scenarios: list[dict] = []
    exhausted_at_budget = False
    for settlement_set, generating_rule in candidate_sets:
        set_accounts = [by_ref[r] for r in sorted(settlement_set) if r in by_ref]
        longest_term = max((a.remaining_term_months or 0) for a in set_accounts) if set_accounts else 0
        for product_code in sorted(route_products(set_accounts)):
            for term in candidate_terms(product_code, longest_term, max_term_extension):
                if len(scenarios) >= budget:
                    exhausted_at_budget = True
                    break
                scenarios.append({
                    "settlement_set": settlement_set, "generating_rule": generating_rule,
                    "product_code": product_code, "term_months": term,
                })
            if exhausted_at_budget:
                break
        if exhausted_at_budget:
            break

    cause = vocab.TERMINATION_BUDGET if exhausted_at_budget else vocab.TERMINATION_EXHAUSTED
    return scenarios, cause
