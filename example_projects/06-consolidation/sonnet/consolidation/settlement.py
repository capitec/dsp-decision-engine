"""Stage 5.3 -- settlement amount derivation (spec 06 §5.3).

"The settlement amount is not the balance." Per settleable account:

    settlement_amount =
          outstanding capital at the assumed settlement date
        + interest accrued since the last statement (per diem x days)
        + fees and charges unpaid at that date
        + early settlement charge, where applicable
        + security release / cancellation cost, where secured
        - unearned fee/premium rebate, where the product provides one

Published for project 11 alongside `inventory.classify_settleability`
(DEPS.md: "11 consumes ... [06] §5.2, §5.3").
"""
from __future__ import annotations

from datetime import date, timedelta

from decider import frame_step

import polars as pl

from consolidation import vocab

# Per-diem: interest accrual since the last statement, capped at 30 days of look-back
# (working-depth stand-in for a real statement-cycle feed).
_ACCRUAL_LOOKBACK_DAYS = 20

# Early settlement charge (§5.3): below the large-agreement threshold, none; at or
# above it, a fixed number of months' interest where notice was not given.
_LARGE_AGREEMENT_THRESHOLD = 250_000.0
_EARLY_SETTLEMENT_MONTHS_CHARGE = {0: 0, 1: 1, 2: 3}  # early_settlement_rule_code -> months

_SETTLEMENT_BUFFER_PCT = 0.015
_SETTLEMENT_BUFFER_CAP = 2_500.0

# Provider payment turnaround + Bank disbursement lag, days (06 §5.3: "1 to 12
# working days"). Internal: instant.
_DISBURSEMENT_LAG_DAYS = 2


def settlement_buffer(settlement_total: float) -> float:
    """1.5% of the settlement total, capped at R2 500 -- a declared tunable (§5.3),
    added once to the *advance*, not per account."""
    return round(min(settlement_total * _SETTLEMENT_BUFFER_PCT, _SETTLEMENT_BUFFER_CAP), 2)


def _assumed_settlement_date(decision_date: date, turnaround_days: int) -> date:
    return decision_date + timedelta(days=turnaround_days + _DISBURSEMENT_LAG_DAYS)


def _settle_one(
    account: dict, turnaround_days: int, early_settlement_rule_code: int, security_release_days: int,
    settleability_code: int, decision_date: date,
) -> dict:
    balance = account.get("balance") or 0.0
    rate = account.get("nominal_annual_rate") or 0.0
    assumed_date = _assumed_settlement_date(decision_date, turnaround_days)
    days = min((assumed_date - decision_date).days, _ACCRUAL_LOOKBACK_DAYS)
    accrued_interest = round(balance * rate / 365.0 * days, 2)

    # `existing_quotation` (§4.3) is flattened onto the account (`quotation_reference`,
    # `quotation_amount`, `quotation_expiry_date`) -- see NOTES.md "Framework friction"
    # for why a nested struct crashes single-record scoring.
    quotation_reference = account.get("quotation_reference")
    quotation_expiry = account.get("quotation_expiry_date")
    if quotation_reference and quotation_expiry is not None and quotation_expiry >= decision_date:
        amount = round(account.get("quotation_amount") or 0.0, 2)
        basis = vocab.BASIS_QUOTED
        component_capital, component_fees, component_esc, component_release, component_rebate = (
            amount, 0.0, 0.0, 0.0, 0.0,
        )
    else:
        fees_unpaid = round((account.get("balance") or 0.0) * 0.0, 2)  # no unpaid-charges feed at this depth
        esc_months = _EARLY_SETTLEMENT_MONTHS_CHARGE.get(early_settlement_rule_code, 0)
        esc = round(balance * rate / 12.0 * esc_months, 2) if balance >= _LARGE_AGREEMENT_THRESHOLD else 0.0
        release_cost = 500.0 if settleability_code == vocab.SETTLE_SECURITY_RELEASE else 0.0
        rebate = round(min(69.0, balance * 0.001), 2)  # unearned monthly service fee, working-depth stand-in
        amount = round(balance + accrued_interest + fees_unpaid + esc + release_cost - rebate, 2)
        basis = vocab.BASIS_DERIVED_INTERNAL if account.get("is_internal") else vocab.BASIS_ESTIMATED
        component_capital, component_fees, component_esc, component_release, component_rebate = (
            balance, fees_unpaid, esc, release_cost, rebate,
        )

    return {
        "account_ref": account.get("account_ref"),
        "settlement_amount": amount,
        "amount_basis_code": basis,
        "assumed_settlement_date": assumed_date.isoformat(),
        "component_capital": component_capital,
        "component_accrued_interest": accrued_interest,
        "component_fees": component_fees,
        "component_early_settlement_charge": component_esc,
        "component_security_release_cost": component_release,
        "component_rebate": component_rebate,
        "quotation_reference": quotation_reference,
        "quotation_expiry_date": quotation_expiry,
        "estimation_tolerance": 0.0 if basis == vocab.BASIS_QUOTED else round(amount * 0.03, 2),
    }


def derive_settlement_amounts(
    accounts: list[dict], settleable_account_refs: list[int], quotation_turnaround_days_list: list[int],
    early_settlement_rule_codes: list[int], security_release_days_list: list[int],
    settleability_account_refs: list[int], settleability_codes: list[int], decision_date: date,
) -> list[dict]:
    """Plain-Python entry point. Only settleable accounts get a settlement amount --
    the rest were never candidates (§5.2)."""
    by_ref = {a.get("account_ref"): a for a in accounts}
    turnaround_by_ref = dict(zip(settleability_account_refs, quotation_turnaround_days_list))
    esc_by_ref = dict(zip(settleability_account_refs, early_settlement_rule_codes))
    release_by_ref = dict(zip(settleability_account_refs, security_release_days_list))
    code_by_ref = dict(zip(settleability_account_refs, settleability_codes))
    out = []
    for ref in settleable_account_refs:
        account = by_ref.get(ref)
        if account is None:
            continue
        out.append(_settle_one(
            account, turnaround_by_ref.get(ref, 5), esc_by_ref.get(ref, 0), release_by_ref.get(ref, 0),
            code_by_ref.get(ref, vocab.SETTLE_QUOTABLE), decision_date,
        ))
    return out


def _process(row: dict) -> dict:
    results = derive_settlement_amounts(
        row.get("accounts") or [], row.get("settleable_account_refs") or [],
        row.get("quotation_turnaround_days_list") or [], row.get("early_settlement_rule_codes") or [],
        row.get("security_release_days_list") or [], row.get("settleability_account_refs") or [],
        row.get("settleability_codes") or [], row["decision_date"],
    )
    total = round(sum(r["settlement_amount"] for r in results), 2)
    return {
        "settlement_account_refs": [r["account_ref"] for r in results],
        "settlement_amounts": [r["settlement_amount"] for r in results],
        "settlement_amount_basis_codes": [r["amount_basis_code"] for r in results],
        "assumed_settlement_dates": [r["assumed_settlement_date"] for r in results],
        "settlement_total": total,
        "settlement_buffer_amount": settlement_buffer(total),
    }


@frame_step(
    reads=["accounts", "settleable_account_refs", "quotation_turnaround_days_list", "early_settlement_rule_codes",
           "security_release_days_list", "settleability_account_refs", "settleability_codes", "decision_date"],
    writes=["settlement_account_refs", "settlement_amounts", "settlement_amount_basis_codes",
            "assumed_settlement_dates", "settlement_total", "settlement_buffer_amount"],
)
def settlement_derivation(df: pl.DataFrame) -> pl.DataFrame:
    results = [
        _process(row) for row in df.select(
            "accounts", "settleable_account_refs", "quotation_turnaround_days_list",
            "early_settlement_rule_codes", "security_release_days_list", "settleability_account_refs",
            "settleability_codes", "decision_date",
        ).to_dicts()
    ]
    return df.with_columns(pl.DataFrame(results))
