"""Stage 5.2 -- obligation inventory and settleability (spec 06 §5.2).

**This stage is where the search space is set** (§5.2): shrinking the raw account
list to the handful that can actually be settled is worth more than any cleverness
in the search itself (§5.5). Nine `settleability_code` values, evaluated per
account in the stated order, first match wins.

Built as a `frame_step`, like `credit_core.obligations` (project 00): the
per-record work is genuinely row-shaped (0..80 accounts, ragged, each
individually classified), and the per-account annotation is emitted as parallel
primitive lists rather than a `list[struct]` column -- the same
`ValueError: cannot parse numpy data type dtype('O') into Polars data type`
project 00 and 02's NOTES.md both document for a `frame_step`'s *output*
(reading a `list[struct]` *input*, as `accounts` is here, is fine).

Published for project 11 (DEPS.md: "11 consumes real references to ... [06]
Obligation inventory and settleability"): `classify_settleability()` is the
plain-Python entry point, callable with no pipeline for a single account list.
"""
from __future__ import annotations

from datetime import date

from decider import frame_step, missing_as, param

import polars as pl

from credit_core.evidence import cell_id as _cell_id
from consolidation import vocab

SETTLEABILITY_TABLE_VERSION = "settleability-2026.09"

# Non-consolidatable account types (§5.2 SETTLE_BLOCKED_POLICY: court-ordered
# maintenance, EAOs, tax debt, municipal, concessionary student loans) -- a
# working-depth subset of the spec's ~45-type x 4-product table (00/02/03's own
# precedent: only the slice's dominant-difficulty table stays at declared size).
_NON_CONSOLIDATABLE_TYPES = frozenset({50, 51, 52, 53})

# Providers that do not issue third-party settlement quotations at all (§5.2
# SETTLE_BLOCKED_PROVIDER: "17 providers, 2.3% of external accounts" in the full
# spec) -- a small representative set, versioned like every other table here.
_NON_QUOTING_PROVIDERS = frozenset({901, 902, 903})

# provider_code -> (quotation_turnaround_days, security_release_days, early_settlement_rule_code).
# ~200 providers in the full spec (06 §6.1); working depth here, default row for
# every provider not listed. Internal accounts (is_internal) never consult this --
# they are always SETTLE_INTERNAL with same-day turnaround.
_PROVIDER_DEFAULT = (5, 10, 1)
_PROVIDER_TABLE = {
    101: (2, 10, 1),   # electronic-exchange provider: fast turnaround
    102: (2, 10, 1),
    201: (10, 10, 2),  # still posts: slow turnaround
}

# Blocked account statuses (§5.2 SETTLE_BLOCKED_STATUS).
_BLOCKED_STATUS_CODES = frozenset({3, 5, 6, 7})  # disputed / handed-over / legal / written-off
_DEBT_REVIEW_STATUS_CODE = 4
_CEDED_STATUS_CODE = 8

# account_type_code -> is this a revolving facility (SETTLE_PARTIAL_REVOLVING candidate)?
_REVOLVING_TYPES = frozenset({10, 11, 12})


def _months_between(d1: date, d2: date) -> int:
    return (d1.year - d2.year) * 12 + (d1.month - d2.month)


def _classify_one(
    account: dict, decision_date: date, recently_opened_months: int, nominated: set, excluded: set,
) -> dict:
    ref = account.get("account_ref")
    account_type = account.get("account_type_code")
    provider = account.get("provider_code")
    is_internal = bool(account.get("is_internal"))
    status = account.get("account_status_code")
    is_disputed = bool(account.get("is_disputed"))
    is_secured = bool(account.get("is_secured"))
    opened = account.get("opened_date")
    security_release_days = 0
    early_settlement_rule = 0
    turnaround = 0

    def result(code: int, rule_id: str) -> dict:
        return {
            "account_ref": ref, "settleability_code": code, "settleability_rule_id": rule_id,
            "quotation_turnaround_days": turnaround, "security_release_days": security_release_days,
            "early_settlement_rule_code": early_settlement_rule,
            "is_mandatory": ref in nominated, "is_client_excluded": ref in excluded,
            "settleability_cell_id": _cell_id("settleability", SETTLEABILITY_TABLE_VERSION, provider, code),
        }

    # First match wins, in the order §5.2 declares (most-blocked / least-informative first).
    if ref is None or account_type is None or (provider is None and not is_internal):
        return result(vocab.SETTLE_UNKNOWN, "CON-SETTLE-UNKNOWN")
    if is_disputed or status in _BLOCKED_STATUS_CODES or status == _DEBT_REVIEW_STATUS_CODE or status == _CEDED_STATUS_CODE:
        return result(vocab.SETTLE_BLOCKED_STATUS, "CON-SETTLE-BLOCKED-STATUS")
    if account_type in _NON_CONSOLIDATABLE_TYPES:
        return result(vocab.SETTLE_BLOCKED_POLICY, "CON-SETTLE-BLOCKED-POLICY-TYPE")
    if opened is not None and _months_between(decision_date, opened) < recently_opened_months:
        return result(vocab.SETTLE_BLOCKED_POLICY, "CON-INT-02")
    if not is_internal and provider in _NON_QUOTING_PROVIDERS:
        return result(vocab.SETTLE_BLOCKED_PROVIDER, "CON-SETTLE-BLOCKED-PROVIDER")

    if is_internal:
        turnaround = 0
        return result(vocab.SETTLE_INTERNAL, "CON-SETTLE-INTERNAL")

    turnaround, security_release_days, early_settlement_rule = _PROVIDER_TABLE.get(provider, _PROVIDER_DEFAULT)

    if is_secured:
        # H4/CON-INT-12: settleable only if the security releases or transfers.
        return result(vocab.SETTLE_SECURITY_RELEASE, "CON-INT-12")
    if account_type in _REVOLVING_TYPES:
        return result(vocab.SETTLE_PARTIAL_REVOLVING, "CON-SETTLE-PARTIAL-REVOLVING")
    # §4.3 `existing_quotation` is flattened onto the account (`quotation_reference`,
    # `quotation_amount`, `quotation_expiry_date`) rather than carried as a nested
    # struct -- see `_make_sample.py`'s `account()` docstring for why.
    if account.get("quotation_reference"):
        return result(vocab.SETTLE_QUOTED, "CON-SETTLE-QUOTED")
    return result(vocab.SETTLE_QUOTABLE, "CON-SETTLE-QUOTABLE")


def classify_settleability(
    accounts: list[dict], decision_date: date, client_nominated_settle: list[int] | None,
    client_excluded_settle: list[int] | None, recently_opened_months: int = 3,
) -> list[dict]:
    """Plain-Python entry point (no pipeline): one classification per account, in
    input order. Published for reuse -- project 07 and 08 want settleability
    later (DEPS.md §13 Q15); project 11 wants it now."""
    nominated = set(client_nominated_settle or [])
    excluded = set(client_excluded_settle or [])
    return [_classify_one(a, decision_date, recently_opened_months, nominated, excluded) for a in accounts]


def _process(row: dict, recently_opened_months: int) -> dict:
    accounts = row.get("accounts") or []
    decision_date = row["decision_date"]
    classifications = classify_settleability(
        accounts, decision_date, row.get("client_nominated_settle"), row.get("client_excluded_settle"),
        recently_opened_months,
    )
    by_code: dict[int, int] = {}
    settleable_refs, mandatory_refs, excluded_refs = [], [], []
    for c in classifications:
        by_code[c["settleability_code"]] = by_code.get(c["settleability_code"], 0) + 1
        if c["is_client_excluded"]:
            excluded_refs.append(c["account_ref"])
            continue
        if c["settleability_code"] in vocab.SETTLEABLE_CODES:
            settleable_refs.append(c["account_ref"])
            if c["is_mandatory"]:
                mandatory_refs.append(c["account_ref"])
    total_obligation = sum((a.get("instalment") or 0.0) for a in accounts)
    settleable_obligation = sum(
        (a.get("instalment") or 0.0) for a in accounts if a.get("account_ref") in settleable_refs
    )
    return {
        "settleability_account_refs": [c["account_ref"] for c in classifications],
        "settleability_codes": [c["settleability_code"] for c in classifications],
        "settleability_rule_ids": [c["settleability_rule_id"] for c in classifications],
        "settleability_cell_ids": [c["settleability_cell_id"] for c in classifications],
        "quotation_turnaround_days_list": [c["quotation_turnaround_days"] for c in classifications],
        "security_release_days_list": [c["security_release_days"] for c in classifications],
        "early_settlement_rule_codes": [c["early_settlement_rule_code"] for c in classifications],
        "settleable_account_refs": settleable_refs,
        "mandatory_account_refs": mandatory_refs,
        "client_excluded_account_refs": excluded_refs,
        "settleable_count": len(settleable_refs),
        "unknown_count": by_code.get(vocab.SETTLE_UNKNOWN, 0),
        "settleable_obligation_proportion": round(settleable_obligation / total_obligation, 4) if total_obligation else 0.0,
        "settleability_table_version": SETTLEABILITY_TABLE_VERSION,
    }


@frame_step(
    reads=["accounts", "decision_date", "client_nominated_settle", "client_excluded_settle"],
    writes=["settleability_account_refs", "settleability_codes", "settleability_rule_ids",
            "settleability_cell_ids", "quotation_turnaround_days_list", "security_release_days_list",
            "early_settlement_rule_codes", "settleable_account_refs", "mandatory_account_refs",
            "client_excluded_account_refs", "settleable_count", "unknown_count",
            "settleable_obligation_proportion", "settleability_table_version"],
)
def settleability(df: pl.DataFrame) -> pl.DataFrame:
    recently_opened_months = 3  # CON-INT-02 default; overridable via params.json (see pipeline.py)
    results = [
        _process(row, recently_opened_months)
        for row in df.select("accounts", "decision_date", "client_nominated_settle", "client_excluded_settle")
        .to_dicts()
    ]
    return df.with_columns(pl.DataFrame(results))
