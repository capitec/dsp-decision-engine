"""Integration: campaign 23 end to end, over `sample_request.json`. Requires `credit_core`
(project 00) and `loan_granting` (project 03) on `PYTHONPATH` -- see SERVE.md.
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
    req["cycle_date"] = datetime.date.fromisoformat(req["cycle_date"])
    return req


def test_sample_request_scores_end_to_end(built, params, record):
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    assert out["is_eligible"] is True
    assert out["leaf_outcome_code"] in (0, 1)
    assert out["overlay_stack_id"].startswith("OS-")


def test_reproduces_the_spec_worked_example(built, params, record):
    """Spec 04 §5.4.1(d): client 8412907 reaches leaf 912, tier B, "Standard top-up, SMS
    responsive"."""
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    assert out["offer_tier_code"] == 2  # tier B
    assert out["reason_label"] == 9103
    assert out["leaf_outcome_code"] == 1  # target


def test_overlay_changes_the_leaf_for_a_client_near_the_threshold(built, params, record):
    """§5.3.4 requirement 2: the unadjusted answer survives alongside the adjusted one,
    through the same pipeline, one params document."""
    near_threshold = {**record, "discretionary_income": 2400.0, "estimated_discretionary_income": 2400.0}
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(near_threshold, params=params)
    assert out["leaf"] != out["unadjusted_leaf"]
    assert out["leaf_outcome_code"] == 0          # adjusted (tightened): do not target
    assert out["unadjusted_leaf_outcome_code"] == 1  # unadjusted (published): target


def test_09_5_15_evidence_spot_check(built, params, record):
    """A subset of the 09 §5.15 contract this project's own evidence record must carry:
    stable logic identity, table version/cell attribution, the overlay stack, reason codes,
    and no "today" (§5.15 items 1-2, 5, 7, 11)."""
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(record, params=params)
    assert out["assignment_id"]                       # item 1: a stable decision identifier
    assert out["leaf"] and out["unadjusted_leaf"]      # item 2: stable, content-derived logic identity
    assert out["appetite_cell_id"].startswith("appetite@")  # item 5: table version and cell, to the cell
    assert out["overlay_stack_id"]                     # item 7: the overlay stack recorded on every decision
    assert out["reason_label"] in [9101, 9102, 9103, 9104, 9201, 9202, 9203, 9204]  # item 11
    assert out["cycle_date"] == record["cycle_date"]   # item 4: resolved from cycle_date, not "today"


def test_batch_and_real_time_agree(built, params, record):
    import polars as pl
    exe = Engine().bind(built, mode="interpreted")
    single = exe.score(record, params=params)
    batch = exe.run(pl.DataFrame([record, record, record]), params=params)
    row = batch.row(1, named=True)
    for field in ("leaf", "unadjusted_leaf", "advertised_amount", "pre_assessed_amount", "overlay_stack_id"):
        assert row[field] == single[field], field


def test_deceased_client_is_ineligible(built, params, record):
    declined = {**record, "is_deceased": True}
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(declined, params=params)
    assert out["is_eligible"] is False


def test_suppressed_client_is_still_evaluated_when_measurement_relevant(built, params, record):
    """§5.2 requirement 3: a measurement-relevant suppression (marketing opt-out) still
    evaluates through the tree and records the path -- suppression is not a silent filter."""
    suppressed = {**record, "marketing_opt_out": True}
    exe = Engine().bind(built, mode="interpreted")
    out = exe.score(suppressed, params=params)
    assert "S07" in out["suppression_codes"]
    assert out["suppressed_absolute"] is False
    assert out["leaf"] is not None  # still evaluated through the tree


def test_decider_build_stages_the_config_version(monkeypatch):
    """The BRIEF's own acceptance test: `decider build` must succeed unchanged."""
    from click.testing import CliRunner
    from decider.cli import cli

    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if not k.upper().startswith("DECIDER_")})
    monkeypatch.setenv("DECIDER_API__MODE", "fused")
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("pipeline", "inference"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.chdir(ROOT)
    result = CliRunner().invoke(cli, ["build"])
    assert result.exit_code == 0, result.output
    assert "built config version 0.1.0" in result.output
