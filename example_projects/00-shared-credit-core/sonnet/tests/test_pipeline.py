"""Integration: the demo pipeline (`pipeline.py`) end to end, over `sample_request.json`.

Exercises the 09-C evidence minimum SCOPE.md requires of every slice: decision id
carried through, stable logic ids (cell_id per table), table version and cell,
overlay stack with unadjusted values, reason codes with registry version, and no
reliance on "today" (every date-sensitive lookup keys off `decision_date`).
"""
import copy
import datetime
import json
from pathlib import Path

import pytest
from decider import Engine
from decider.steps.tables import DecisionTableConfig

import pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rate_card():
    return DecisionTableConfig.load(str(ROOT / "configs" / "0.1.0" / "rate_card_flex_loan.json"))


@pytest.fixture(scope="module")
def built(rate_card):
    return pipeline.build(rate_card)


@pytest.fixture(scope="module")
def params():
    return json.loads((ROOT / "configs" / "0.1.0" / "params.json").read_text())


@pytest.fixture
def record():
    req = json.loads((ROOT / "sample_request.json").read_text())
    req["decision_date"] = datetime.date.fromisoformat(req["decision_date"])
    for acc in req["bureau_accounts"] + req["internal_accounts"]:
        acc["opened_date"] = datetime.date.fromisoformat(acc["opened_date"])
    return req


def test_the_sample_request_scores_end_to_end(built, params, record):
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    assert out["outcome_code"] in (pipeline.OUTCOME_APPROVE, pipeline.OUTCOME_APPROVE_WITH_CONDITIONS,
                                    pipeline.OUTCOME_REFER, pipeline.OUTCOME_DECLINE)
    assert out["instalment"] > 0
    assert out["nominal_annual_rate"] > 0


def test_decision_id_is_carried_through_unchanged(built, params, record):
    """09 §5.15 item 1: present on every record, unchanged."""
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    assert out["decision_id"] == record["decision_id"]


def test_every_table_read_carries_its_version_and_cell(built, params, record):
    """09 §5.15 item 5: which version, which cell -- for every table this slice reads."""
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    for cell_field, version_field in [
        ("tax_table_cell_id", None), ("statutory_norm_cell_id", None), ("internal_norm_cell_id", None),
        ("calibration_cell_id", "calibration_version"), ("risk_grade_cell_id", "risk_grade_version"),
        ("rate_card_cell_id", "rate_card_version"),
    ]:
        assert out[cell_field] is not None, cell_field
        if version_field:
            assert out[version_field] is not None, version_field


def test_the_overlay_stack_is_recorded_with_the_unadjusted_value(built, params, record):
    """09 §5.15 item 7: overlay stack, its effect, and the unadjusted input it consumed."""
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    assert out["adjustment_set_id"] == pipeline.ADJUSTMENT_SET_ID
    assert out["adjustments_applied"] == ["ADJ-2026-001", "ADJ-2026-002"]
    assert out["score_unadjusted"] != out["score"]
    assert out["score"] == out["score_unadjusted"] - 18.0 - 5.0


def test_the_stack_can_be_disabled_through_the_same_pipeline(built, params, record):
    """Acceptance §10 item 8: stack-off run, no separate implementation."""
    exe = Engine().bind(built, mode="interpreted")
    on = exe.score(record, params=params)
    off_params = copy.deepcopy(params)
    off_params["credit_core_demo"]["apply_score_adjustments"]["adjustment_stack_enabled"] = False
    off = exe.score(record, params=off_params)
    assert off["score"] == off["score_unadjusted"]
    assert on["score"] != off["score"]
    assert on["score_unadjusted"] == off["score_unadjusted"]


def test_replay_reproduces_the_decision_to_the_cent(built, params, record):
    """Acceptance §10 item 6, and 09 §5.15 item 4 (no "today"): re-running the same
    decision_date and inputs must reproduce the same output, bit for bit."""
    exe = Engine().bind(built, mode="interpreted")
    first = exe.score(record, params=params)
    second = exe.score(record, params=params)
    assert first == second


def test_decline_reasons_carry_the_registry_version(built, params, record):
    """09 §5.15 item 11: declared reason codes, registry version, severity order, primary."""
    exe = Engine().bind(built, mode="interpreted")
    # Force an eligibility decline so a reason actually fires.
    declined_record = {**record, "is_sanctioned": True}
    out = exe.score(declined_record, params=params)
    assert out["reason_registry_version"] == pipeline.REASON_REGISTRY.version
    assert out["primary_reason_code"] in out["decline_reason_codes"]
    assert out["outcome_code"] == pipeline.OUTCOME_DECLINE


def test_decider_build_stages_the_config_version(monkeypatch):
    """The BRIEF's own acceptance test: `decider build` must succeed unchanged.

    Mirrors `tests/cli/test_cli.py`'s pattern in the `decider` repo: drive the
    real CLI in-process with `CliRunner`, from this project's own directory.
    """
    import os
    import sys

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
