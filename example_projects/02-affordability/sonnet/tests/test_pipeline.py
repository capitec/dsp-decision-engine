"""Integration: the full assessment pipeline (`pipeline.py`) end to end, over
`sample_request.json`. Exercises the 09-C evidence minimum SCOPE.md requires of every
slice: decision id carried through, stable logic ids (cell_id per table), table version
and cell, overlay stack with unadjusted values, reason codes with registry version, and
no reliance on "today"."""
import copy
import datetime
import json
from pathlib import Path

import pytest
from decider import Engine
from decider.steps.tables import DecisionTableConfig  # noqa: F401  (imported for parity with 00's test style)

import pipeline
from assessment import modes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built():
    return pipeline.build()


@pytest.fixture(scope="module")
def params(built):
    return built.parameters().defaults()


@pytest.fixture
def record():
    req = json.loads((ROOT / "sample_request.json").read_text())
    req["decision_date"] = datetime.date.fromisoformat(req["decision_date"])
    req["bureau_as_of_date"] = datetime.date.fromisoformat(req["bureau_as_of_date"])
    for key in ("applicant1_bureau_accounts", "applicant1_internal_accounts",
                "applicant2_bureau_accounts", "applicant2_internal_accounts"):
        for a in req.get(key, []):
            if a.get("opened_date"):
                a["opened_date"] = datetime.date.fromisoformat(a["opened_date"])
    return req


def test_the_sample_request_scores_end_to_end(built, params, record):
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    assert out["affordability_verdict_code"] in (1, 2, 3, 4)
    assert out["max_affordable_instalment"] >= 0
    assert out["discretionary_income_after"] is not None  # sample request supplies proposed_instalment


def test_every_table_read_carries_its_version_and_cell(built, params, record):
    """09 §5.15 item 5: which version, which cell -- for every table this slice reads."""
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    for cell_field in ("statutory_norm_cell_id", "internal_norm_cell_id",
                       "buffer_grid_cell_id", "residual_floor_cell_id"):
        assert out[cell_field] is not None, cell_field
    assert out["norm_table_version"] is not None


def test_the_overlay_stack_is_recorded_with_the_unadjusted_value(built, params, record):
    """09 §5.15 item 7: overlay stack, its effect, and the unadjusted value it consumed."""
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    assert out["adjustment_set_id"] == pipeline.ADJUSTMENT_SET_ID
    assert out["adjustments_applied"] == ["ADJ-02-2026-001"]  # product 10 is in scope
    assert out["max_affordable_instalment"] != out["max_affordable_instalment_unadjusted"]
    assert out["max_affordable_instalment"] == round(out["max_affordable_instalment_unadjusted"] * 0.90, 2)


def test_the_stack_can_be_disabled_through_the_same_pipeline(built, params, record):
    """Acceptance §10 item 9: stack-off run, through the same implementation."""
    exe = Engine().bind(built, mode="interpreted")
    on = exe.score(record, params=params)
    off_params = copy.deepcopy(params)
    off_params["affordability_assessment"]["capacity"]["overlay"]["apply_max_affordable_instalment_adjustments"][
        "adjustment_stack_enabled"] = False
    off = exe.score(record, params=off_params)
    assert off["max_affordable_instalment"] == off["max_affordable_instalment_unadjusted"]
    assert on["max_affordable_instalment"] != off["max_affordable_instalment"]
    assert on["max_affordable_instalment_unadjusted"] == off["max_affordable_instalment_unadjusted"]


def test_replay_reproduces_the_decision_to_the_cent(built, params, record):
    """Acceptance §10 item 5 / 09 §5.15 item 4 (no "today"): re-running the same
    decision_date and inputs must reproduce the same output, bit for bit."""
    exe = Engine().bind(built, mode="interpreted")
    first = exe.score(record, params=params)
    second = exe.score(record, params=params)
    assert first == second


def test_decline_reasons_carry_the_registry_version_when_they_fire():
    """09 §5.15 item 11: declared reason codes, registry version, severity order, primary."""
    from assessment import verdict as _verdict
    exe = Engine().bind(pipeline._reasons_unit(), mode="interpreted")
    out = exe.score({"evidence_sufficiency_code": _verdict.EVIDENCE_BUREAU_STALE,
                      "affordability_verdict_code": _verdict.INDETERMINATE})
    assert out["reason_registry_version"] == pipeline.REASON_REGISTRY.version
    assert out["primary_reason_code"] in out["decline_reason_codes"]
    assert pipeline.R_BUREAU_STALE in out["decline_reason_codes"]


def test_indeterminate_is_never_returned_as_fail_across_every_evidence_gap(built, params, record):
    """Acceptance §10 item 7: one test per `evidence_sufficiency_code` cause."""
    from assessment import verdict as _verdict
    exe = Engine().bind(built, mode="interpreted")
    scenarios = {
        "stale bureau": {"bureau_as_of_date": "2020-01-01"},
        "unestablished applicant income": {"applicant1_payslip_income": None, "applicant1_declared_income": None},
    }
    for name, overrides in scenarios.items():
        rec = copy.deepcopy(record)
        for k, v in overrides.items():
            if v is None:
                rec.pop(k, None)
            elif k == "bureau_as_of_date":
                rec[k] = datetime.date.fromisoformat(v)
        out = exe.score(rec, params=params)
        assert out["affordability_verdict_code"] == _verdict.INDETERMINATE, name
        assert out["affordability_verdict_code"] != _verdict.FAIL, name
        assert out["evidence_sufficiency_code"] != _verdict.EVIDENCE_OK, name


@pytest.mark.parametrize("mode_code,mode_name", [
    (modes.NEW_APPLICATION, "new_application"),
    (modes.LIMIT_INCREASE, "limit_increase"),
    (modes.ARRANGEMENT, "arrangement"),
    (modes.SCENARIO, "scenario"),
])
def test_every_consuming_mode_scores_through_the_one_pipeline(built, params, record, mode_code, mode_name):
    """02 §5.8: one arithmetic path, four modes -- one acceptance test per consuming
    project's mode (SCOPE.md)."""
    exe = Engine().bind(built, mode="interpreted")
    rec = copy.deepcopy(record)
    rec["assessment_mode_code"] = mode_code
    out = exe.score(rec, params=params)
    assert out["affordability_verdict_code"] in (1, 2, 3, 4), mode_name
    assert out["minimum_income_tier"] == params["affordability_assessment"]["evidence"]["minimum_income_tier"][
        f"minimum_tier_{mode_name}"]


def test_decider_build_stages_the_config_version(monkeypatch):
    """The BRIEF's own acceptance test: `decider build` must succeed unchanged."""
    import os
    import sys

    import decider.settings as settings_module
    from click.testing import CliRunner
    from decider.cli import cli

    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if not k.upper().startswith("DECIDER_")})
    monkeypatch.setenv("DECIDER_API__MODE", "fused")
    monkeypatch.setattr(settings_module, "settings", settings_module.settings)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("pipeline", "inference"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.chdir(ROOT)
    result = CliRunner().invoke(cli, ["build"])
    assert result.exit_code == 0, result.output
    assert "built config version 0.1.0" in result.output
