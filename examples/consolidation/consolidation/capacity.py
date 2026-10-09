"""The affordability assessment's capacity half, scored once per candidate debt set.

Income, tax and living expenses do not depend on which accounts are settled, so the pipeline derives them once
(`assessment.evidence_unit()`) and hands every scenario the same figures. Only the obligations, and everything that
follows from them, are recomputed here: `assessment.capacity_unit()` in `SCENARIO` mode, where an account carrying a
settlement quote has been paid off and leaves the list.
"""
from __future__ import annotations

import copy
import functools
import json
from datetime import date

from assessment import modes
from assessment.capacity import build_buffer_grid
from assessment.documents import shipped
from assessment.pipeline import capacity_unit
from decider import ConfigurableStep, allow_fallback, each, flow, step

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


@functools.cache
def _compiled(adjustments: str, reason_codes: str) -> tuple:
    unit = capacity_unit(ConfigurableStep.load(adjustments), ConfigurableStep.load(reason_codes))
    # ponytail: interpreted, a fused compile is ten seconds of numba to save 0.15 ms a scenario
    exe = unit.compile("interpreted")
    on = unit.parameters().defaults()
    off = copy.deepcopy(on)
    off["capacity"]["overlay"]["apply_max_affordable_instalment_adjustments"]["adjustment_stack_enabled"] = False
    return exe, {True: on, False: off}  # a params dict is read once per identity, so these are built once


class Capacity:
    """`capacity_unit()` compiled, scored with a debt set and, optionally, the instalment being proposed.

    The compiled unit is an executable, not data a step could be handed, so it is built from the two documents the
    first time they are used and kept for the life of the process, whichever step asks. `adjustments` and
    `reason_codes` are the JSON of the `capacity_unit` documents; a changed document is a different unit.

    Example::

        capacity = Capacity(*capacity_documents())
        capacity(case, accounts, {})["max_affordable_instalment"]
    """

    def __init__(self, adjustments: str, reason_codes: str):
        self._exe, self._params = _compiled(adjustments, reason_codes)

    def __call__(self, case: dict, accounts: list[dict], settled: dict[int, float], proposed_instalment: float | None = None,
                 stack_enabled: bool = True) -> dict:
        bureau, internal = as_assessment_accounts(accounts, settled)
        record = {**case, "assessment_mode_code": modes.SCENARIO, "applicant1_bureau_accounts": bureau,
                  "applicant1_internal_accounts": internal}
        if proposed_instalment is not None:
            record["proposed_instalment"] = proposed_instalment
        return self._exe.score(record, self._params[stack_enabled])


def _behaviour(document) -> str:
    # Governance fields (owner, approval, rationale, review date, descriptions) change no decision, so they stay out of
    # what the steps hold and editing them is not a change to the step.
    data = json.loads(document.model_dump_json())
    for overlay in data.get("adjustments", ()):
        overlay.update(owner="", approval_reference="", rationale="", review_date=overlay["effective_from"])
    for code in data.get("codes", ()):
        code.update(description="", is_regulatory=False)
    return json.dumps(data, sort_keys=True)


def capacity_documents(capacity_adjustments=None, reason_codes=None) -> tuple[str, str]:
    """The JSON of the documents `Capacity` is built from, without their governance fields; each defaults to the one the
    affordability example ships."""
    return tuple(_behaviour(shipped(name) if document is None else document)
                 for name, document in (("capacity_adjustments", capacity_adjustments), ("reason_codes", reason_codes)))


def case_record(decision_date: date, product_code: int, channel_code: int, segment_code: float, court_ordered_deductions: float,
                invariants: dict, affordability_buffer_applied: float) -> dict:
    """What `capacity_unit()` reads besides the debts: the invariants, and the one figure that depends on the product,
    the buffer."""
    return {"decision_date": decision_date, "product_code": product_code, "channel_code": channel_code,
            "segment_code": segment_code, "court_ordered_deductions": court_ordered_deductions, "is_joint_application": False,
            "affordability_buffer_applied": affordability_buffer_applied, **{k: invariants[k] for k in INVARIANTS}}


@allow_fallback
@step(output="buffer_probes")
def buffer_probes(product_policies: dict, risk_grade: float) -> list[dict]:
    """One probe per product the search can price: the keys the buffer grid is read with."""
    return [{"product_code": int(code), "risk_grade": int(risk_grade)} for code in product_policies]


def buffer_unit():
    """The affordability buffer for each product's scenarios: the assessment's own grid, read once per product.

    Writes `scenario_buffers`, a probe per product with its `affordability_buffer_applied`.
    """
    return flow(buffer_probes, each("buffer_probes", flow(build_buffer_grid(), name="grid"), name="scenario_buffers",
                                    output="scenario_buffers"), name="buffers")
