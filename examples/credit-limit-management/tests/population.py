"""A synthetic book for the cycle tests: the sample request, perturbed per account."""
from __future__ import annotations

import copy
import json
import random
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_LISTS = ("applicant1_bureau_accounts", "applicant1_internal_accounts",
                 "applicant2_bureau_accounts", "applicant2_internal_accounts")
DATES = ("decision_date", "bureau_as_of_date", "salary_deposit_as_of_date", "last_change_date")


def parse_dates(record: dict) -> dict:
    """What the handler does to a JSON request before the pipeline sees it."""
    record = copy.deepcopy(record)
    for field in DATES:
        if isinstance(record.get(field), str):
            record[field] = date.fromisoformat(record[field])
    for accounts in ACCOUNT_LISTS:
        for account in record.get(accounts, []):
            account["opened_date"] = date.fromisoformat(account["opened_date"])
    return record


def sample_request() -> dict:
    return json.loads((ROOT / "sample_request.json").read_text())


def make_book(n: int, seed: int = 7) -> pl.DataFrame:
    """`n` accounts; a client holds two accounts in every pair. Roughly a third are eligible for an increase."""
    rng, base = random.Random(seed), parse_dates(sample_request())
    rows = []
    for i in range(n):
        limit = float(rng.choice([5_000, 10_000, 15_000, 25_000, 40_000]))
        utilisation, spend = rng.uniform(0.0, 1.1), rng.uniform(1_000, 9_000)
        income = rng.uniform(10_000, 60_000)
        arrears = 0 if rng.random() > 0.08 else rng.choice([1, 2])
        rows.append(base | {
            "account_id": 900_000_000 + i, "client_id": 500_000_000 + i // 2,
            "product_code": rng.choice([20, 21]), "current_limit": limit,
            "months_on_book": rng.randint(6, 120),
            "cycle_balances": [round(limit * utilisation * rng.uniform(0.8, 1.2), 2) for _ in range(6)],
            "cycle_limits": [limit] * 6,
            "cycle_purchase_values": [round(spend * rng.uniform(0.7, 1.3), 2) for _ in range(6)],
            "worst_arrears_months_24": rng.choice([0, 0, 0, 1, 2, 3]),
            "months_since_last_arrears": rng.randint(0, 48),
            "payment_to_balance_ratio_6m": round(rng.uniform(0.05, 0.9), 3),
            "over_limit_cycles_12m": rng.choice([0, 0, 0, 1, 2]),
            "cash_withdrawal_ratio_12m": round(rng.uniform(0.0, 0.5), 3),
            "bureau_enquiry_velocity_6m": round(rng.uniform(0, 8), 2),
            "worst_arrears_months_now": arrears, "worst_arrears_months_6": arrears,
            "consent_automatic_increase": rng.random() > 0.1,
            "declared_gross_income_on_file": round(income, 2), "declared_net_income_on_file": round(income * 0.8, 2),
            "total_exposure_other": round(rng.uniform(0, 40_000), 2),
            "verified_salary_present_3m": rng.random() > 0.25,
            "salary_deposit_variability_pct": round(rng.uniform(0.0, 0.35), 3),
            "verified_salary_amount": round(income * rng.uniform(0.9, 1.05), 2),
            "declared_income_raw": round(income, 2), "declared_income_age_days": rng.randint(10, 1200),
        })
    return pl.DataFrame(rows)
