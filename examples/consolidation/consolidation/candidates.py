"""Which settlement sets are worth pricing, in what order.

A client with 18 settleable accounts has 262 143 sets, each carried by up to four products at several terms. The
search never enumerates them. Credit Risk Policy's ordering rules each put the accounts in a preferred order, and the
prefixes of that order (settle the top 1, top 2, ...) are the candidates, beside the client's nominated set and the
full set. The empty set, doing nothing, is not a candidate: the baseline is the comparator every scenario is measured
against. The order is total and uses nothing but the inputs, so the same client always gets the same candidates in
the same order.
"""
from __future__ import annotations

from typing import Callable

from decikit import ParamTable, param_table

from consolidation import vocab
from consolidation.inventory import num

ORDERING_RULE_SET_VERSION = "ordering-rules-2026.09"

# The rules in force, in sequence: retire a rule by removing its row, reorder by moving it.
ORDERING_RULES = param_table({"rule": int}, default=[{"rule": n} for n in range(1, 9)])


def _rate(a: dict) -> float:
    return -num(a, "nominal_annual_rate")


def _relief_per_rand(a: dict) -> float:
    return -num(a, "instalment") / a["settlement_amount"] if a["settlement_amount"] else 0.0


def _longest_first(a: dict) -> float:
    return -(num(a, "remaining_term_months") or float("inf"))  # a revolving account has no term to run out


# (rule number, name, sort key). Every key ends in the account reference, so a tie resolves the same way every time.
RULES = (
    (1, "H1_highest_rate_first", lambda a: (_rate(a), a["account_ref"])),
    (2, "H2_most_relief_per_rand", lambda a: (_relief_per_rand(a), a["account_ref"])),
    (3, "H3_shortest_remaining_term_last", lambda a: (_longest_first(a), a["account_ref"])),
    (4, "H4_secured_last", lambda a: (a["settleability_code"] == vocab.SECURITY_RELEASE, _rate(a), a["account_ref"])),
    (5, "H5_provider_in_arrears_first", lambda a: (not a["provider_in_arrears"], _rate(a), a["account_ref"])),
    (6, "H6_quotation_held_first", lambda a: (not a["quotation_held"], _rate(a), a["account_ref"])),
    (7, "H7_whole_provider_relationship_first",
     lambda a: (not a["provider_fully_settleable"], a.get("provider_code"), _rate(a), a["account_ref"])),
    (8, "H8_highest_reaccumulation_risk_first",
     lambda a: (a["account_type_code"] not in vocab.REVOLVING_TYPES, _rate(a), a["account_ref"])),
)


def rule(number: int) -> tuple[str, Callable[[dict], tuple]]:
    """`(name, sort key)` of the ordering rule with this number."""
    return next((name, key) for n, name, key in RULES if n == number)


def with_provider_facts(settleable: list[dict], all_accounts: list[dict]) -> list[dict]:
    """The settleable accounts, each knowing whether its provider has an account in arrears and whether every account
    the client holds there could be settled (rules H5 and H7)."""
    settleable_refs = {a["account_ref"] for a in settleable}
    in_arrears = {a.get("provider_code") for a in all_accounts if num(a, "months_in_arrears") > 0}
    kept_open = {a.get("provider_code") for a in all_accounts if a["account_ref"] not in settleable_refs}
    return [{**a, "provider_in_arrears": a.get("provider_code") in in_arrears,
             "provider_fully_settleable": a.get("provider_code") not in kept_open} for a in settleable]


def candidate_sets(settleable: list[dict], mandatory: list[int], rules: ParamTable,
                   max_prefix: int) -> list[tuple[tuple[int, ...], str]]:
    """Every candidate settlement set with the rule that produced it, in generation order, without duplicates.

    A set that omits a mandatory account is never generated.
    """
    must = frozenset(mandatory)
    seen: set[frozenset] = set()
    out: list[tuple[tuple[int, ...], str]] = []

    def add(refs, rule: str) -> None:
        refs = frozenset(refs) | must
        if refs not in seen:
            seen.add(refs)
            out.append((tuple(sorted(refs)), rule))

    if must:
        add(must, "client_nominated")
    for number in rules.rule.tolist():
        name, key = rule(number)
        ordered = [a["account_ref"] for a in sorted(settleable, key=key)]
        for k in range(1, min(len(ordered), max_prefix) + 1):
            add(ordered[:k], name)
    add((a["account_ref"] for a in settleable), "full_settleable_set")
    return out
