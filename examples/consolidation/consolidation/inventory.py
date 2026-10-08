"""The client's accounts, and which of them can enter a settlement set at all.

Every account is classified by the first rule that matches (`settleability_code`, 7 first, 0 last), and the
classification records the rule and the table cell behind it, so "why was my store card not even considered" has an
answer. The provider facts live in the `settleability_rules` table, a config document Credit Risk Policy edits.
"""
from __future__ import annotations

from datetime import date
from typing import TypedDict

from decider import (
    DecisionTableConfig,
    ParamTable,
    allow_fallback,
    each,
    flow,
    missing_as,
    param,
    param_table,
    step,
)

from consolidation import vocab


class Account(TypedDict, total=False):
    account_ref: int
    account_type_code: int
    provider_code: int
    is_internal: bool
    balance: float
    credit_limit: float
    instalment: float
    nominal_annual_rate: float
    remaining_term_months: float
    months_in_arrears: int
    opened_date: date
    is_secured: bool
    account_status_code: int
    is_disputed: bool
    days_since_statement: int
    fees_unpaid: float
    unearned_rebate: float
    quotation_reference: str
    quotation_amount: float
    quotation_expiry_date: date


def num(account: dict, field: str) -> float:
    """A numeric field of an account, 0 where the account does not carry it."""
    return account.get(field) or 0.0


@allow_fallback
def prepared_accounts(
    accounts: list[Account], decision_date: date,
    client_nominated_settle: list[int] = missing_as([]), client_excluded_settle: list[int] = missing_as([]),
) -> list[dict]:
    """The accounts with the facts the per-account rules need: age, whether a quotation is held and unexpired, and
    what the client asked for."""
    out = []
    for a in accounts:
        opened = a.get("opened_date")
        months = (decision_date.year - opened.year) * 12 + decision_date.month - opened.month if opened else 999
        expiry = a.get("quotation_expiry_date")
        out.append({**a, "months_since_opened": months,
                    "quotation_held": bool(a.get("quotation_reference")) and expiry is not None and expiry >= decision_date,
                    "is_nominated": a["account_ref"] in client_nominated_settle,
                    "is_excluded": a["account_ref"] in client_excluded_settle})
    return out


@allow_fallback
def provider_known(settleability_cell_id: str | None) -> bool:
    return settleability_cell_id is not None


NON_CONSOLIDATABLE = param_table({"account_type_code": int}, default=[
    {"account_type_code": 50},  # court-ordered maintenance
    {"account_type_code": 51},  # emoluments attachment order
    {"account_type_code": 52},  # tax debt
    {"account_type_code": 53},  # municipal account
])


@allow_fallback
@step(outputs=("settleability_code", "settleability_rule_id"))
def settleability(
    account_type_code: int | None, provider_known: bool, quotes_to_third_parties: int, is_internal: bool,
    months_since_opened: float, quotation_held: bool, is_disputed: bool = missing_as(False),
    account_status_code: int = missing_as(0), is_secured: bool = missing_as(False),
    recently_opened_months: int = param(3, ge=0, le=12), non_consolidatable: ParamTable = NON_CONSOLIDATABLE,
) -> tuple[int, str]:
    """The first rule that matches, in the order unknown, status, policy, provider, security, revolving, quote."""
    if account_type_code is None or not provider_known:
        return vocab.UNKNOWN, "CON-SETTLE-UNKNOWN"
    if is_disputed:
        return vocab.BLOCKED_BY_STATUS, "CON-INT-11"
    if account_status_code != 0:
        return vocab.BLOCKED_BY_STATUS, "CON-SETTLE-STATUS"
    if account_type_code in non_consolidatable.account_type_code:
        return vocab.BLOCKED_BY_POLICY, "CON-SETTLE-ACCOUNT-TYPE"
    if months_since_opened < recently_opened_months:
        return vocab.BLOCKED_BY_POLICY, "CON-INT-02"
    if not quotes_to_third_parties:
        return vocab.BLOCKED_BY_PROVIDER, "CON-SETTLE-PROVIDER"
    if is_secured:
        return vocab.SECURITY_RELEASE, "CON-INT-12"
    if account_type_code in vocab.REVOLVING_TYPES:
        return vocab.PARTIAL_REVOLVING, "CON-SETTLE-REVOLVING"
    if is_internal:
        return vocab.INTERNAL, "CON-SETTLE-INTERNAL"
    if quotation_held:
        return vocab.QUOTED, "CON-SETTLE-QUOTED"
    return vocab.QUOTABLE, "CON-SETTLE-QUOTABLE"


def classify_unit(settleability_rules: DecisionTableConfig):
    """Prepare the accounts, then classify each one: writes `classified_accounts`."""
    per_account = flow(settleability_rules.relabel(writes={"cell_id": "settleability_cell_id"}), provider_known,
                       settleability, name="account")
    return flow(prepared_accounts, each("prepared_accounts", per_account, name="classify", output="classified_accounts"),
                inventory_summary, name="inventory")


@allow_fallback
@step(outputs=("settleable_account_refs", "mandatory_account_refs", "excluded_account_refs", "nomination_conflicts",
                "settleable_count", "unknown_count", "settleable_obligation_proportion"))
def inventory_summary(classified_accounts: list[dict]) -> tuple[list[int], list[int], list[int], list[int], int, int, float]:
    """Who can be settled. A client exclusion removes an account from the search (and is recorded); a nominated
    account that cannot be settled is a conflict for the consultant to resolve, never silently dropped."""
    settleable = [a["account_ref"] for a in classified_accounts
                  if a["settleability_code"] in vocab.SETTLEABLE and not a["is_excluded"]]
    nominated = {a["account_ref"] for a in classified_accounts if a["is_nominated"]}
    mandatory = [r for r in settleable if r in nominated]
    conflicts = sorted(nominated - set(settleable))
    total = sum(num(a, "instalment") for a in classified_accounts)
    share = sum(num(a, "instalment") for a in classified_accounts if a["account_ref"] in settleable) / total if total else 0.0
    return (settleable, mandatory, [a["account_ref"] for a in classified_accounts if a["is_excluded"]], conflicts,
            len(settleable), sum(a["settleability_code"] == vocab.UNKNOWN for a in classified_accounts), round(share, 4))
