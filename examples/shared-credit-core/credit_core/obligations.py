"""Existing debt: two account lists (bureau and internal) merged, each account treated by its type.

The answer has two shapes: monthly totals, and parallel per-account lists (index-aligned with each other).
"""
from __future__ import annotations

from datetime import date
from typing import NamedTuple, TypedDict

from decider import allow_fallback, missing_as, step

# obligation_treatment_code
USE_STATED_INSTALMENT = 1
IMPUTE_PCT_OF_LIMIT = 2
IMPUTE_PCT_OF_BALANCE = 3
EXCLUDE = 4
EXCLUDE_IF_SETTLEMENT_IN_FLIGHT = 5
USE_SETTLEMENT_QUOTE = 6
IMPUTE_MINIMUM_PAYMENT_PCT = 7
EXCLUDE_IF_CLOSED = 8
EXCLUDE_INTERNAL_BEING_CONSOLIDATED = 9
AMORTISE_BALANCE_OVER_STANDARD_TERM = 10

# account_type_code -> how its monthly cost is established; a type not listed uses its stated instalment.
TREATMENT_MATRIX = (
    (1, USE_STATED_INSTALMENT),                  # Flex Loan
    (2, USE_STATED_INSTALMENT),                  # vehicle finance
    (3, USE_STATED_INSTALMENT),                  # home loan
    (10, IMPUTE_PCT_OF_LIMIT),                   # credit card
    (11, IMPUTE_MINIMUM_PAYMENT_PCT),            # store card
    (12, IMPUTE_PCT_OF_BALANCE),                 # overdraft
    (20, USE_STATED_INSTALMENT),                 # personal loan at another bank
    (21, AMORTISE_BALANCE_OVER_STANDARD_TERM),   # facility with no clear term
    (30, EXCLUDE_IF_SETTLEMENT_IN_FLIGHT),       # account under settlement negotiation
    (31, EXCLUDE),                               # paid up
    (40, EXCLUDE_INTERNAL_BEING_CONSOLIDATED),   # internal account being consolidated
)

_LIMIT_PCT = 0.03
_BALANCE_PCT = 0.05
_MINIMUM_PAYMENT_PCT = 0.025
_STANDARD_TERM_MONTHS = 36


class Account(TypedDict, total=False):
    account_type_code: int
    balance: float
    limit: float
    instalment: float
    months_in_arrears: int
    opened_date: date
    closed: bool
    is_internal: bool
    settlement_quote: float


class Obligations(NamedTuple):
    existing_obligations: float
    obligations_internal: float
    obligations_external: float
    worst_arrears_months: int
    accounts_in_arrears_count: int
    total_exposure: float
    revolving_utilisation: float
    obligation_account_type_codes: list[int]
    obligation_treatment_codes: list[int]
    obligation_monthly_amounts: list[float]
    obligation_is_internal: list[bool]


def _treat(account: Account, matrix: dict[int, int]) -> tuple[float, int]:
    """(monthly cost, treatment actually used) for one account."""
    if account.get("closed"):
        return 0.0, EXCLUDE_IF_CLOSED
    behaviour = matrix.get(account.get("account_type_code"), USE_STATED_INSTALMENT)
    quote = account.get("settlement_quote")
    if behaviour == EXCLUDE:
        return 0.0, EXCLUDE
    if behaviour == EXCLUDE_INTERNAL_BEING_CONSOLIDATED and account.get("is_internal"):
        return 0.0, EXCLUDE_INTERNAL_BEING_CONSOLIDATED
    if behaviour == EXCLUDE_IF_SETTLEMENT_IN_FLIGHT and quote is not None:
        return 0.0, EXCLUDE_IF_SETTLEMENT_IN_FLIGHT
    if quote is not None:
        return round(quote / _STANDARD_TERM_MONTHS, 2), USE_SETTLEMENT_QUOTE
    balance = account.get("balance") or 0.0
    if behaviour == IMPUTE_PCT_OF_LIMIT:
        return round((account.get("limit") or 0.0) * _LIMIT_PCT, 2), behaviour
    if behaviour == IMPUTE_PCT_OF_BALANCE:
        return round(balance * _BALANCE_PCT, 2), behaviour
    if behaviour == IMPUTE_MINIMUM_PAYMENT_PCT:
        return round(balance * _MINIMUM_PAYMENT_PCT, 2), behaviour
    if behaviour == AMORTISE_BALANCE_OVER_STANDARD_TERM:
        return round(balance / _STANDARD_TERM_MONTHS, 2), behaviour
    return round(account.get("instalment") or 0.0, 2), USE_STATED_INSTALMENT


def _key(account: Account) -> tuple:
    return account.get("account_type_code"), account.get("opened_date"), round(account.get("balance") or 0.0, 2)


def assess(bureau_accounts: list[Account], internal_accounts: list[Account]) -> Obligations:
    """Merge the lists and treat each account. A bureau account that matches an internal one by type, opening
    date and balance is the same account seen twice, so the internal record (the authoritative one) wins.

    Example: `assess([], [{"account_type_code": 1, "instalment": 900.0}]).existing_obligations == 900.0`
    """
    internal_keys = {_key(a) for a in internal_accounts}
    accounts = [*internal_accounts, *(a for a in bureau_accounts if _key(a) not in internal_keys)]
    matrix = dict(TREATMENT_MATRIX)
    treated = [_treat(a, matrix) for a in accounts]
    internal_total = sum(cost for a, (cost, _) in zip(accounts, treated) if a.get("is_internal"))
    external_total = sum(cost for a, (cost, _) in zip(accounts, treated) if not a.get("is_internal"))
    arrears = [a.get("months_in_arrears") or 0 for a in accounts]
    revolving = [a for a in accounts
                 if matrix.get(a.get("account_type_code")) in (IMPUTE_PCT_OF_LIMIT, IMPUTE_MINIMUM_PAYMENT_PCT)]
    revolving_limit = sum(a.get("limit") or 0.0 for a in revolving)
    revolving_balance = sum(a.get("balance") or 0.0 for a in revolving)
    return Obligations(
        existing_obligations=round(internal_total + external_total, 2),
        obligations_internal=round(internal_total, 2),
        obligations_external=round(external_total, 2),
        worst_arrears_months=max(arrears, default=0),
        accounts_in_arrears_count=sum(1 for m in arrears if m > 0),
        total_exposure=round(sum(a.get("balance") or 0.0 for a in accounts), 2),
        revolving_utilisation=round(revolving_balance / revolving_limit, 4) if revolving_limit else 0.0,
        obligation_account_type_codes=[a.get("account_type_code") for a in accounts],
        obligation_treatment_codes=[treatment for _, treatment in treated],
        obligation_monthly_amounts=[cost for cost, _ in treated],
        obligation_is_internal=[bool(a.get("is_internal")) for a in accounts],
    )


@allow_fallback
@step(outputs=Obligations._fields)
def obligations(
    bureau_accounts: list[Account] = missing_as([]), internal_accounts: list[Account] = missing_as([]),
) -> tuple[float, float, float, int, int, float, float, list[int], list[int], list[float], list[bool]]:
    return assess(bureau_accounts, internal_accounts)
