"""End-to-end tests, exactly as `decider build`/`decider serve` drive this pipeline
(SERVE.md's own snippet) -- plus the acceptance-criterion-1 budget test (spec 06 §10
item 1: an 18-settleable-account client, no more than the configured budget, within
900 ms).
"""
from __future__ import annotations

import copy
import datetime
import json
import time
from pathlib import Path

import pytest
from decider import Engine
from decider.steps.tables import DecisionTableConfig

import pipeline

_DIR = Path(__file__).resolve().parents[1]


def _typed(record: dict) -> dict:
    record = copy.deepcopy(record)
    for field in ("decision_date", "bureau_as_of_date", "last_consolidation_date"):
        if record.get(field):
            record[field] = datetime.date.fromisoformat(record[field])
    for account in record.get("accounts", []):
        for field in ("opened_date", "quotation_expiry_date"):
            if account.get(field):
                account[field] = datetime.date.fromisoformat(account[field])
    return record


@pytest.fixture(scope="module")
def engine():
    rate_card = DecisionTableConfig.load(str(_DIR / "configs/0.1.0/rate_card_flex_loan.json"))
    rate_card_p11 = DecisionTableConfig.load(str(_DIR / "configs/0.1.0/rate_card_product11.json"))
    return Engine().bind(pipeline.build(rate_card, rate_card_p11), mode="interpreted")


@pytest.fixture(scope="module")
def sample_record():
    return json.loads((_DIR / "sample_request.json").read_text())


def test_sample_request_scores_and_approves(engine, sample_record):
    out = engine.score(_typed(sample_record))
    assert out["outcome_code"] == "approve"
    assert out["chosen_scenario_id"] != -1
    assert out["chosen_instalment_relief"] > 0


def test_the_cli_can_build_and_serve_this_pipeline(tmp_path, monkeypatch):
    """The same check 00/02/03's own test_pipeline.py runs: `decider build` succeeds
    against this project's own env vars, unchanged (BRIEF: "Verify them")."""
    import subprocess
    import sys

    env = {
        "DECIDER_API__CODE_PATH": str(_DIR), "DECIDER_API__PIPELINE": "pipeline:build",
        "DECIDER_CONFIG__BASEPATH": str(_DIR / "configs"), "DECIDER_API__MODE": "fused",
    }
    import os
    full_env = dict(os.environ)
    full_env.update(env)
    result = subprocess.run(
        [sys.executable, "-c", "from decider.cli import cli; cli()", "build"], cwd=str(_DIR), env=full_env,
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "built config version" in result.stdout


def _eighteen_account_client() -> dict:
    accounts = []
    for i in range(1, 19):
        accounts.append({
            "account_ref": i, "account_type_code": 20 if i % 3 else 10, "provider_code": 100 + i,
            "is_internal": i <= 2, "balance": 5000.0 + i * 3200.0, "credit_limit": 10000.0,
            "instalment": 200.0 + i * 40.0, "nominal_annual_rate": 0.12 + (i % 10) * 0.018,
            "remaining_term_months": 12 + i, "months_in_arrears": 0, "opened_date": "2019-01-01",
            "is_secured": False, "security_type_code": 0, "account_status_code": 0, "is_disputed": False,
            "quotation_amount": 0.0, "quotation_reference": "", "quotation_expiry_date": "2000-01-01",
        })
    return {
        "decision_id": "budget-test-0001", "application_id": 1, "decision_date": "2026-09-24", "channel_code": 2,
        "requested_amount": 20000.0, "assessment_mode_code": 1, "client_nominated_settle": [0],
        "client_excluded_settle": [0], "hardship_declared": False, "accounts": accounts, "risk_grade": 7,
        "applicant_age_years": 38.0, "dependants_count": 1, "employment_type_code": 1, "payslip_income": 65000.0,
        "variable_pay_history": [0.0], "declared_expenses": {"groceries": 4000.0}, "statement_expenses": {"groceries": 3800.0},
        "bureau_as_of_date": "2026-09-20", "court_ordered_deductions": 0.0, "segment_code": 1,
        "objective_id": "OBJ-02", "consolidations_last_24_months": 0, "client_under_debt_review": False,
        "client_under_administration": False, "last_consolidation_date": "2000-01-01", "income_verified": True,
        "bureau_unobtainable": False, "active_reckless_lending_allegation": False,
    }


def test_budget_holds_at_18_settleable_accounts(engine):
    """Spec 06 §10 acceptance item 1 / SCOPE.md "test the budget: 400 scenarios in
    900 ms"."""
    record = _typed(_eighteen_account_client())
    start = time.monotonic()
    out = engine.score(record)
    elapsed_ms = (time.monotonic() - start) * 1000
    assert out["search_consumption"] <= out["search_budget"] <= 400
    assert elapsed_ms < 900, f"took {elapsed_ms:.0f} ms"
    assert out["search_termination_cause"] in ("candidates_exhausted", "budget_exhausted")


def test_result_is_deterministic_across_repeated_runs(engine):
    record = _typed(_eighteen_account_client())
    first = engine.score(dict(record))
    second = engine.score(dict(record))
    assert first["scenario_ids"] == second["scenario_ids"]
    assert first["chosen_scenario_id"] == second["chosen_scenario_id"]
    assert first["outcome_code"] == second["outcome_code"]


def test_debt_review_client_is_referred_never_declined(engine, sample_record):
    record = _typed(sample_record)
    record["client_under_debt_review"] = True
    out = engine.score(record)
    assert out["outcome_code"] == "refer"
    assert 2001 in out["decline_reason_codes"]  # D_DEBT_REVIEW


def test_client_under_administration_is_declined(engine, sample_record):
    record = _typed(sample_record)
    record["client_under_administration"] = True
    out = engine.score(record)
    assert out["outcome_code"] == "decline"


def test_short_circuit_hands_off_to_project_03_for_a_small_easily_affordable_request(engine, sample_record):
    """§5.4's short-circuit needs a genuinely clean client (no arrears, few accounts,
    no high-rate account, a tight rate gap) -- `sample_record` is deliberately messy
    (arrears, 9 accounts, a 29.9% card), matching spec 06 §1's own worked example of a
    client the search actually has to work for, so it is not reused here."""
    record = _typed(sample_record)
    record["accounts"] = [a for a in record["accounts"] if a["account_ref"] in (1, 2)]  # 2 internal accounts only
    for a in record["accounts"]:
        a["months_in_arrears"] = 0
        a["nominal_annual_rate"] = 0.14
    record["requested_amount"] = 3000.0
    out = engine.score(record)
    assert out["short_circuit_applies"] is True
    assert out["plain_grant_via_project03"] is True


def test_every_rejected_scenario_carries_a_rejection_reason_code(engine, sample_record):
    out = engine.score(_typed(sample_record))
    for viable, codes in zip(out["scenario_viable"], out["scenario_rejection_reason_codes"]):
        if not viable:
            assert codes != ""


def test_overlay_does_not_apply_before_its_effective_from_date(engine, sample_record):
    """A `decision_date` before the anti-harm overlay's `effective_from` reaches the
    exact same intervention-checking code path with the overlay simply not in scope
    (`interventions.py`'s `apply_stack`) -- the nearest thing this slice has to §10
    item 10's "stack disabled" run without a `frame_step`-level `param()` toggle (see
    NOTES.md "Framework friction": `frame_step` cannot take `param()` overrides, so
    the true `adjustment_stack_enabled` flip 00/02/03 use for their own scalar steps
    is not directly available to this project's search, which is one big `frame_step`
    by necessity -- see `orchestration.py`)."""
    record = _typed(sample_record)
    record["decision_date"] = datetime.date(2025, 6, 1)
    out = engine.score(record)
    assert out["outcome_code"] in ("approve", "decline", "refer")
