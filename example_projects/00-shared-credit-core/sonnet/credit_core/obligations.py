"""`core.obligations` -- existing debt obligations (spec 00 §6.4; addendum A10).

Consumes **two** ragged account lists -- bureau-reported and internally
held -- de-duplicates them, and returns both a scalar aggregate and a
per-account annotation (addendum item 10; 00 §6.4 "Hard part": "one
capability, two shapes of answer"). Ten treatment behaviours decide how
each account contributes to `existing_obligations` (addendum: "ten... not
five").

Built as a `frame_step`: the per-record work is genuinely row-shaped (a
variable-length list of accounts, each individually annotated), which a
scalar `Step` cannot express, and `frame_step` is exactly decider's escape
hatch for that (BRIEF: "frame_step").

# ponytail: the merge loop is pure Python per record (O(accounts^2) for
# the dedup match), not a vectorised polars join -- correct and testable
# at the 0..95-account volumes this slice runs, but not tuned for the
# library's 15 ms p99 budget at 14 M-record batch scale (00 §9). Upgrade
# path: an anti-join on (account_type_code, opened_date) in polars if this
# capability shows up in a batch profile.
"""
from __future__ import annotations

from decider import frame_step

import polars as pl

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

# account_type_code -> treatment behaviour (~15 types at working depth; spec is ~45).
TREATMENT_MATRIX = {
    1: USE_STATED_INSTALMENT,             # Flex Loan
    2: USE_STATED_INSTALMENT,             # vehicle finance
    3: USE_STATED_INSTALMENT,             # home loan
    10: IMPUTE_PCT_OF_LIMIT,              # credit card
    11: IMPUTE_MINIMUM_PAYMENT_PCT,       # store card
    12: IMPUTE_PCT_OF_BALANCE,            # overdraft
    20: USE_STATED_INSTALMENT,            # personal loan (other bank)
    21: AMORTISE_BALANCE_OVER_STANDARD_TERM,  # informal / unclear-term facility
    30: EXCLUDE_IF_SETTLEMENT_IN_FLIGHT,  # account under settlement negotiation
    31: EXCLUDE,                          # paid-up / zero-balance
    40: EXCLUDE_INTERNAL_BEING_CONSOLIDATED,  # internal account flagged for this consolidation
}

_IMPUTE_LIMIT_PCT = 0.03      # revolving: 3% of limit
_IMPUTE_BALANCE_PCT = 0.05    # overdraft-shaped: 5% of balance
_MIN_PAYMENT_PCT = 0.025      # store card minimum payment
_STANDARD_TERM_MONTHS = 36


def _treat_one(account: dict) -> tuple[float, int]:
    """One account -> (monthly obligation, treatment code actually used)."""
    if account.get("closed"):
        return 0.0, EXCLUDE_IF_CLOSED
    behaviour = TREATMENT_MATRIX.get(account.get("account_type_code"), USE_STATED_INSTALMENT)
    balance = account.get("balance") or 0.0
    limit = account.get("limit") or 0.0
    stated = account.get("instalment") or 0.0
    settlement_quote = account.get("settlement_quote")
    settlement_in_flight = settlement_quote is not None

    if behaviour == EXCLUDE:
        return 0.0, EXCLUDE
    if behaviour == EXCLUDE_INTERNAL_BEING_CONSOLIDATED and account.get("is_internal"):
        return 0.0, EXCLUDE_INTERNAL_BEING_CONSOLIDATED
    if behaviour == EXCLUDE_IF_SETTLEMENT_IN_FLIGHT and settlement_in_flight:
        return 0.0, EXCLUDE_IF_SETTLEMENT_IN_FLIGHT
    if settlement_in_flight and behaviour != EXCLUDE_IF_SETTLEMENT_IN_FLIGHT:
        return round(settlement_quote / _STANDARD_TERM_MONTHS, 2), USE_SETTLEMENT_QUOTE
    if behaviour == IMPUTE_PCT_OF_LIMIT:
        return round(limit * _IMPUTE_LIMIT_PCT, 2), IMPUTE_PCT_OF_LIMIT
    if behaviour == IMPUTE_PCT_OF_BALANCE:
        return round(balance * _IMPUTE_BALANCE_PCT, 2), IMPUTE_PCT_OF_BALANCE
    if behaviour == IMPUTE_MINIMUM_PAYMENT_PCT:
        return round(balance * _MIN_PAYMENT_PCT, 2), IMPUTE_MINIMUM_PAYMENT_PCT
    if behaviour == AMORTISE_BALANCE_OVER_STANDARD_TERM:
        return round(balance / _STANDARD_TERM_MONTHS, 2), AMORTISE_BALANCE_OVER_STANDARD_TERM
    return round(stated, 2), USE_STATED_INSTALMENT


def _dedup_key(account: dict) -> tuple:
    return (account.get("account_type_code"), account.get("opened_date"), round(account.get("balance") or 0.0, 2))


def _merge_accounts(bureau_accounts: list[dict], internal_accounts: list[dict]) -> list[dict]:
    """Internal records are authoritative; a bureau record matching one by (type, opened date,
    balance) is the same account seen twice and is dropped, not double-counted."""
    internal_keys = {_dedup_key(a) for a in internal_accounts}
    merged = list(internal_accounts)
    for b in bureau_accounts:
        if _dedup_key(b) not in internal_keys:
            merged.append(b)
    return merged


def _process(bureau_accounts, internal_accounts) -> dict:
    accounts = _merge_accounts(bureau_accounts or [], internal_accounts or [])
    # Per-account annotation as four parallel, index-aligned lists of primitives, not one
    # list[struct] column -- see the frame_step / list[struct] note in this module's
    # docstring and NOTES.md "Framework friction": decider's Engine can serialize a
    # ragged list[float]/list[int] output fine, but a list-of-dicts terminal output from a
    # `frame_step` fails at result-materialisation time (`ValueError: cannot parse numpy
    # data type dtype('O') into Polars data type`), because `FrameStep` outputs are
    # declared `Any`-typed and the generic output path round-trips through numpy, which
    # can't box a list of Python dicts. Four parallel lists carry the identical
    # information and survive that path.
    account_type_codes, treatment_codes, monthly_obligations, is_internal_flags = [], [], [], []
    internal_total = external_total = total_exposure = 0.0
    worst_arrears = 0
    arrears_count = 0
    revolving_balance = revolving_limit = 0.0
    for a in accounts:
        obligation, treatment = _treat_one(a)
        is_internal = bool(a.get("is_internal"))
        if is_internal:
            internal_total += obligation
        else:
            external_total += obligation
        total_exposure += a.get("balance") or 0.0
        months_in_arrears = a.get("months_in_arrears") or 0
        worst_arrears = max(worst_arrears, months_in_arrears)
        if months_in_arrears > 0:
            arrears_count += 1
        if TREATMENT_MATRIX.get(a.get("account_type_code")) in (IMPUTE_PCT_OF_LIMIT, IMPUTE_MINIMUM_PAYMENT_PCT):
            revolving_balance += a.get("balance") or 0.0
            revolving_limit += a.get("limit") or 0.0
        account_type_codes.append(a.get("account_type_code"))
        treatment_codes.append(treatment)
        monthly_obligations.append(obligation)
        is_internal_flags.append(is_internal)
    return {
        "existing_obligations": round(internal_total + external_total, 2),
        "obligations_internal": round(internal_total, 2),
        "obligations_external": round(external_total, 2),
        "worst_arrears_months": worst_arrears,
        "accounts_in_arrears_count": arrears_count,
        "total_exposure": round(total_exposure, 2),
        "revolving_utilisation": round(revolving_balance / revolving_limit, 4) if revolving_limit else 0.0,
        "obligation_account_type_codes": account_type_codes,
        "obligation_treatment_codes": treatment_codes,
        "obligation_monthly_amounts": monthly_obligations,
        "obligation_is_internal": is_internal_flags,
    }


@frame_step(reads=["bureau_accounts", "internal_accounts"],
            writes=["existing_obligations", "obligations_internal", "obligations_external", "worst_arrears_months",
                    "accounts_in_arrears_count", "total_exposure", "revolving_utilisation",
                    "obligation_account_type_codes", "obligation_treatment_codes", "obligation_monthly_amounts",
                    "obligation_is_internal"])
def obligations(df: pl.DataFrame) -> pl.DataFrame:
    results = [
        _process(row.get("bureau_accounts") or [], row.get("internal_accounts") or [])
        for row in df.select("bureau_accounts", "internal_accounts").to_dicts()
    ]
    out = pl.DataFrame(results)
    return df.with_columns(out)
