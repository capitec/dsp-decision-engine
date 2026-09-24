"""A synthetic population of accounts for allocation/batch tests, built by
perturbing `sample_request.json` -- not a second request schema."""
from __future__ import annotations

import random

from limit_mgmt.vocab import PRODUCT_ACCESS_FACILITY, PRODUCT_EVERYDAY_CARD


def make_population(base_record: dict, n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        r = dict(base_record)
        r["account_id"] = 900_000_000 + i
        r["client_id"] = 500_000_000 + (i // 2)  # some clients hold two accounts
        r["product_code"] = rng.choice([PRODUCT_EVERYDAY_CARD, PRODUCT_ACCESS_FACILITY])
        r["current_limit"] = float(rng.choice([5_000, 10_000, 15_000, 25_000, 40_000]))
        r["months_on_book"] = rng.randint(6, 120)
        mean_util = rng.uniform(0.0, 1.1)
        r["mean_utilisation_6m"] = round(min(mean_util, 1.0), 4)
        # `revolving_utilisation_6m`/`_3m` are *produced* by the pipeline
        # (`population.utilisation_means`, from `cycle_balances`/`cycle_limits`) -- not
        # set here, or `Engine.run` refuses the frame ("'revolving_utilisation_6m' is
        # produced by this pipeline and is also a column of the input frame").
        r["cycle_balances"] = [round(r["current_limit"] * mean_util * rng.uniform(0.8, 1.2), 2) for _ in range(6)]
        r["cycle_limits"] = [r["current_limit"]] * 6
        spend = rng.uniform(500, 8000)
        r["cycle_purchase_values"] = [round(spend * rng.uniform(0.7, 1.3), 2) for _ in range(6)]
        r["worst_arrears_months_24"] = rng.choice([0, 0, 0, 1, 2, 3])
        r["months_since_last_arrears"] = rng.randint(0, 48)
        r["payment_to_balance_ratio_6m"] = round(rng.uniform(0.05, 0.9), 3)
        r["over_limit_cycles_12m"] = rng.choice([0, 0, 0, 1, 2])
        r["cash_withdrawal_ratio_12m"] = round(rng.uniform(0.0, 0.5), 3)
        r["bureau_enquiry_velocity_6m"] = round(rng.uniform(0, 8), 2)
        r["worst_arrears_months_now"] = 0 if rng.random() > 0.08 else rng.choice([1, 2])
        r["worst_arrears_months_6"] = r["worst_arrears_months_now"]
        r["consent_automatic_increase"] = rng.random() > 0.1
        r["consent_withdrawn"] = False
        r["fraud_marker"] = False
        r["deceased_marker"] = False
        r["debt_review_active"] = rng.random() < 0.02
        r["treatment_suspension_active"] = rng.random() < 0.02
        r["declared_gross_income_on_file"] = round(rng.uniform(8_000, 60_000), 2)
        r["declared_net_income_on_file"] = round(r["declared_gross_income_on_file"] * 0.8, 2)
        r["total_exposure_other"] = round(rng.uniform(0, 40_000), 2)
        r["verified_salary_present_3m"] = rng.random() > 0.25
        r["salary_deposit_variability_pct"] = round(rng.uniform(0.0, 0.35), 3)
        r["declared_income_age_days"] = rng.randint(10, 1200)
        r["verified_salary_amount"] = r["declared_gross_income_on_file"] * rng.uniform(0.9, 1.05)
        r["declared_income_raw"] = r["declared_gross_income_on_file"]
        out.append(r)
    return out
