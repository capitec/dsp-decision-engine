"""The affordability assessment's capacity half, scored once per candidate debt set.

Income, tax and living expenses do not depend on which accounts are settled, so the pipeline derives them once
(`assessment.evidence_unit()`) and hands every scenario the same figures. Only the obligations, and everything that
follows from them, are recomputed here: `assessment.capacity_unit()` in `SCENARIO` mode, where an account carrying a
settlement quote has been paid off and leaves the list.
"""
from __future__ import annotations

import copy
from datetime import date

from assessment import modes
from assessment.capacity import build_buffer_grid
from assessment.pipeline import capacity_unit

from consolidation import vocab
from consolidation.inventory import num

# The evidence outputs `capacity_unit()` reads.
INVARIANTS = ("net_monthly_income", "living_expenses", "income_verification_tier", "applicant_income_evidence_gap",
              "bureau_is_stale", "minimum_income_tier", "residual_floor_amount")


def debt_service_ratio(obligations: float, net_monthly_income: float) -> float:
    """Obligations as a share of net income; with no income the ratio is 1, so a ceiling always fails."""
    return round(obligations / net_monthly_income, 4) if net_monthly_income > 0 else 1.0


def as_assessment_accounts(accounts: list[dict], settled: dict[int, float]) -> tuple[list[dict], list[dict]]:
    """`(bureau, internal)` account lists in `credit_core`'s shape; a settled account carries its settlement as a quote."""
    bureau, internal = [], []
    for a in accounts:
        record = {"account_type_code": a["account_type_code"], "balance": num(a, "balance"), "limit": num(a, "credit_limit"),
                  "instalment": num(a, "instalment"), "months_in_arrears": int(num(a, "months_in_arrears")),
                  "closed": a.get("account_status_code") == vocab.CLOSED_STATUS, "is_internal": bool(a["is_internal"])}
        if a.get("opened_date") is not None:
            record["opened_date"] = a["opened_date"]
        if a["account_ref"] in settled:
            record["settlement_quote"] = settled[a["account_ref"]]
        (internal if a["is_internal"] else bureau).append(record)
    return bureau, internal


class Capacity:
    """`capacity_unit()` compiled once. Call it with a debt set and, optionally, the instalment being proposed."""

    def __init__(self):
        unit = capacity_unit()
        # ponytail: interpreted, a fused compile is ten seconds of numba to save 0.15 ms a scenario
        self._exe = unit.compile("interpreted")
        on = unit.parameters().defaults()
        off = copy.deepcopy(on)
        off["capacity"]["overlay"]["apply_max_affordable_instalment_adjustments"]["adjustment_stack_enabled"] = False
        self._params = {True: on, False: off}  # a params dict is read once per identity, so these are built once

    def __call__(self, case: dict, accounts: list[dict], settled: dict[int, float], proposed_instalment: float | None = None,
                 stack_enabled: bool = True) -> dict:
        bureau, internal = as_assessment_accounts(accounts, settled)
        record = {**case, "assessment_mode_code": modes.SCENARIO, "applicant1_bureau_accounts": bureau,
                  "applicant1_internal_accounts": internal}
        if proposed_instalment is not None:
            record["proposed_instalment"] = proposed_instalment
        return self._exe.score(record, self._params[stack_enabled])


def case_record(decision_date: date, product_code: int, channel_code: int, segment_code: float, court_ordered_deductions: float,
                invariants: dict, affordability_buffer_applied: float) -> dict:
    """What `capacity_unit()` reads besides the debts: the invariants, and the one figure that depends on the product,
    the buffer."""
    return {"decision_date": decision_date, "product_code": product_code, "channel_code": channel_code,
            "segment_code": segment_code, "court_ordered_deductions": court_ordered_deductions, "is_joint_application": False,
            "affordability_buffer_applied": affordability_buffer_applied, **{k: invariants[k] for k in INVARIANTS}}


class Buffers:
    """The assessment's buffer grid, read for each product a scenario is priced under."""

    def __init__(self):
        self._exe = build_buffer_grid().compile("interpreted")
        self._seen: dict[tuple[int, int], float] = {}

    def __call__(self, product_code: int, risk_grade: int) -> float:
        key = (product_code, risk_grade)
        if key not in self._seen:
            row = self._exe.score({"product_code": product_code, "risk_grade": risk_grade})
            self._seen[key] = row["affordability_buffer_applied"]
        return self._seen[key]
