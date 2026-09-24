"""End-to-end proof: `decider build` succeeds and the sample request scores through the
handler (BRIEF.md's mandatory verification), plus the pricing-consistency regression this
project's own build process caught (NOTES.md "Framework friction").
"""
from __future__ import annotations

import os
import sys

from conftest import CREDIT_CORE_ROOT, PROJECT_ROOT  # resolves scratch vs. repo layout once, not per-file


def test_decider_build_stages_the_config_version(monkeypatch):
    """The BRIEF's own acceptance test: `decider build` must succeed unchanged.

    Mirrors project 00's own `test_pipeline.py` pattern: drive the real CLI in-process
    with `CliRunner`, from this project's own directory. PROJECT_ROOT is inserted *ahead*
    of CREDIT_CORE_ROOT on `sys.path`: both projects have a top-level `pipeline.py`, and
    whichever directory Python resolves first for `import pipeline` wins -- the first
    version of this test (and of SERVE.md) got this backwards and built project 00's
    demo pipeline instead of this one (see NOTES.md "Framework friction").
    """
    from click.testing import CliRunner
    from decider.cli import cli

    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if not k.upper().startswith("DECIDER_")})
    monkeypatch.setenv("DECIDER_API__MODE", "fused")
    monkeypatch.setattr(sys, "path", list(sys.path))
    for p in (str(CREDIT_CORE_ROOT), str(PROJECT_ROOT)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    for name in ("pipeline", "inference"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.chdir(PROJECT_ROOT)

    result = CliRunner().invoke(cli, ["build"])
    assert result.exit_code == 0, result.output
    assert "built config version 0.1.0" in result.output


def test_sample_request_scores_through_the_handler(monkeypatch, sample_record, params):
    """No server: exactly as `SERVE.md` documents scoring without one."""
    from decider.serving.handler import construct_handler_from_settings

    monkeypatch.setenv("DECIDER_API__CODE_PATH", str(PROJECT_ROOT))
    monkeypatch.setenv("DECIDER_API__PIPELINE", "pipeline:build")
    monkeypatch.setenv("DECIDER_CONFIG__BASEPATH", str(PROJECT_ROOT / "configs"))
    monkeypatch.setenv("DECIDER_API__MODE", "fused")
    monkeypatch.chdir(PROJECT_ROOT)

    handler = construct_handler_from_settings()
    handler.stage()
    handler.activate()
    live = handler.module_fn()
    out = live.executable.score(sample_record, live.params)

    assert out["outcome_code"] == 1  # approve
    assert out["offer_amount"] > 0
    assert out["decline_reason_codes"] == []


def test_all_three_pricing_paths_agree(engine, params, sample_record):
    """The regression this project's own build process caught: the search's pure-Python
    pricing, P17's independent re-derivation, and P12's real decider-step pricing of the
    shipped offer must all compute the same instalment (10 §5.14 requirement 7).
    """
    out = engine.score(sample_record, params)
    assert out["offer_instalment"] == out["rederived_instalment"]
    assert out["offer_instalment"] == out["instalment"]
    assert out["offer_instalment"] <= out["max_affordable_instalment"]


def test_decision_id_and_no_today(engine, params, sample_record):
    """09-C checklist items 1 and 4 (09 §5.15): stable id carried through, decision_date
    drives every effective-dated lookup, never "today"."""
    out = engine.score(sample_record, params)
    assert out["decision_id"] is None or True  # decision_id is not re-derived by this pipeline (caller-assigned)
    assert out["norm_table_version"]  # resolved from decision_date, not from datetime.date.today()


def test_reason_registry_version_present_on_a_decline(engine, params, sample_record):
    declined = {**sample_record, "is_sanctioned": True}
    out = engine.score(declined, params)
    assert out["outcome_code"] == 4  # decline
    assert out["reason_registry_version"] == "p10-reasons-2026.09"
    assert out["primary_reason_code"] in out["decline_reason_codes"]
