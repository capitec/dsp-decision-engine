"""Integration: the full granting-and-pricing pipeline end to end, over
`sample_request.json`. Requires `credit_core` (project 00) and `assessment`
(project 02) on `PYTHONPATH` -- see SERVE.md.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

import pytest
from decider import Engine
from decider.steps.tables import DecisionTableConfig

import pipeline

ROOT = Path(__file__).resolve().parents[1]
DATE_FIELDS = ("decision_date", "bureau_as_of_date")
ACCOUNT_LIST_FIELDS = (
    "bureau_accounts", "internal_accounts", "applicant2_bureau_accounts", "applicant2_internal_accounts",
)


@pytest.fixture(scope="module")
def rate_card():
    return DecisionTableConfig.load(str(ROOT / "configs" / "0.1.0" / "rate_card_flex_loan.json"))


@pytest.fixture(scope="module")
def built(rate_card):
    return pipeline.build(rate_card)


@pytest.fixture(scope="module")
def params(built):
    return built.parameters().defaults()


@pytest.fixture
def record():
    req = json.loads((ROOT / "sample_request.json").read_text())
    for field in DATE_FIELDS:
        req[field] = datetime.date.fromisoformat(req[field])
    for field in ACCOUNT_LIST_FIELDS:
        for a in req[field]:
            a["opened_date"] = datetime.date.fromisoformat(a["opened_date"])
    return req


def test_the_sample_request_scores_end_to_end(built, params, record):
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    assert out["outcome_code"] in (1, 2, 3, 4)
    assert out["is_eligible"] is True
    assert out["risk_grade"] in range(1, 13)
    assert len(out["offer_term_months"]) <= 9


def test_an_approved_application_has_exactly_one_recommended_offer(built, params, record):
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    assert out["outcome_code"] == 1  # APPROVE
    assert sum(out["offer_is_recommended"]) == 1
    assert out["has_any_offer"] is True
    assert out["final_validation_passed"] is True


def test_every_offer_independently_re_validates(built, params, record):
    """Spec 03 §5.8 requirement 7 / §10 item 2: the recommended offer's final
    validation must pass -- a fresh re-derivation, not a re-run of the search."""
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    assert out["final_validation_passed"] is True
    assert out["final_validation_failures"] == []


def test_the_cap_waterfall_chain_is_retrievable_for_every_ceiling(built, params, record):
    """09 §5.15 item 15: cap chains recorded, not just the final value."""
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    assert out["amount_cap_binding_rule_id"] is not None
    assert len(out["waterfall_rule_ids"]) == 52
    assert len(out["waterfall_rule_status"]) == 52


def test_ineligible_application_declines_without_computing_an_offer(built, params, record):
    declined = {**record, "deceased_flag": True}
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(declined, params=params)
    assert out["outcome_code"] == 4  # DECLINE
    assert out["is_eligible"] is False
    assert 1150 in out["decline_reason_codes"]  # R_DECEASED_ESTATE
    assert out["primary_reason_code"] in out["decline_reason_codes"]


def test_fraud_decline_short_circuits_to_decline(built, params, record):
    declined = {**record, "fraud_verdict_code": 3}
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(declined, params=params)
    assert out["outcome_code"] == 4
    assert 1210 in out["decline_reason_codes"]


def test_the_recommendation_objective_changes_the_recommended_offer_without_a_rebuild(built, record):
    """Spec 03 §10 item 7 / change scenario 4: switching the objective is a params
    change, not a code change."""
    exe = Engine().bind(built, mode="interpreted")
    defaults = built.parameters().defaults()

    def with_objective(objective: str) -> dict:
        import copy
        p = copy.deepcopy(defaults)
        p["loan_granting"]["recommendation_objective"]["recommendation_objective"] = objective
        return p

    largest = exe.score(record, params=with_objective("largest_amount"))
    cheapest = exe.score(record, params=with_objective("lowest_total_cost"))
    assert largest["recommended_term"] != cheapest["recommended_term"] or \
        largest["recommended_amount"] != cheapest["recommended_amount"]


def test_the_overlay_stack_can_be_run_disabled_through_the_same_pipeline(built, record):
    """00 §7.6 / 09 §5.14.3: the stack-off run is the same implementation, one param flip."""
    exe = Engine().bind(built, mode="interpreted")
    defaults = built.parameters().defaults()
    on = exe.score(record, params=defaults)
    import copy
    off_params = copy.deepcopy(defaults)
    for node in off_params["loan_granting"]["scoring"].values():
        if isinstance(node, dict) and "adjustment_stack_enabled" in node:
            node["adjustment_stack_enabled"] = False
    off = exe.score(record, params=off_params)
    assert on["score_unadjusted"] == off["score_unadjusted"] == off["score"]


def test_batch_and_real_time_agree(built, params, record):
    """Spec 03 §10 item 11 (a small-scale proof; SCOPE.md skips the 14 M-record timing)."""
    import polars as pl
    exe = Engine().bind(built, mode="interpreted")
    single = exe.score(record, params=params)
    batch = exe.run(pl.DataFrame([record, record, record]), params=params)
    row = batch.row(1, named=True)
    for field in ("outcome_code", "risk_grade", "amount_cap", "recommended_amount", "recommended_instalment"):
        assert row[field] == single[field], field


def test_decider_build_stages_the_config_version(monkeypatch):
    """The BRIEF's own acceptance test: `decider build` must succeed unchanged."""
    import decider.settings as settings_module
    from click.testing import CliRunner
    from decider.cli import cli

    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if not k.upper().startswith("DECIDER_")})
    monkeypatch.setenv("DECIDER_API__MODE", "interpreted")
    monkeypatch.setattr(settings_module, "settings", settings_module.settings)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("pipeline", "inference"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.chdir(ROOT)
    result = CliRunner().invoke(cli, ["build"])
    assert result.exit_code == 0, result.output
    assert "built config version 0.1.0" in result.output
