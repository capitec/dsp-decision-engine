import datetime
import json
import os
import subprocess
from pathlib import Path

import pytest
from decider import Engine
from decider.steps.trees import TreeConfig

from fraud_interdiction import backtest
import pipeline

_HERE = Path(__file__).resolve().parents[1]
_CONFIGS = _HERE / "configs" / "0.1.0"


def _load_pipeline():
    live = TreeConfig.load(json.loads((_CONFIGS / "live_rules.json").read_text()))
    shadow = TreeConfig.load(json.loads((_CONFIGS / "shadow_rules.json").read_text()))
    base = TreeConfig.load(json.loads((_CONFIGS / "overlay_base_rules.json").read_text()))
    return pipeline.build(live, shadow, base)


def _sample_record() -> dict:
    record = json.loads((_HERE / "sample_request.json").read_text())
    record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
    return record


@pytest.fixture(scope="module")
def built_pipeline():
    return _load_pipeline()


def test_sample_request_scores_a_hold_for_review_decline(built_pipeline):
    """The shipped sample request: a first payment to a brand-new, high-mule-band beneficiary,
    moments after account creation, on the web channel -- multiple MS and AT rules fire."""
    out = Engine().bind(built_pipeline).score(_sample_record(), {})
    assert out["action_code"] in (40, 50)  # hold_for_review or decline
    assert out["action_source_rule_id"]
    assert len(out["fired_rule_ids"]) > 0
    assert out["primary_reason_code"] in out["decline_reason_codes"]
    # shadow rules must never appear where the action-visible fields are read from
    assert set(out["shadow_fired_rule_ids"]).isdisjoint(out["fired_rule_ids"])


def test_hard_block_floors_the_action_regardless_of_firing_set():
    """§5.7: a hard block sets a floor even when no rule would otherwise ask for that much."""
    built = _load_pipeline()
    record = _sample_record()
    record.update(amount=10.0, beneficiary_first_payment=False, beneficiary_age_hours=5000.0,
                   beneficiary_bank_mule_rate_band=1, device_change_hours=5000.0, sim_change_hours=5000.0,
                   session_ip_distance_band=0, model_score=100.0, cop_name_match_band=3,
                   velocity_distinct_counterparties_24h=0.0, client_prior_confirmed_fraud_count=0.0,
                   account_frozen=True)
    out = Engine().bind(built).score(record, {})
    assert out["action_code"] == 50  # decline floor, even though nothing else fired that high
    assert out["action_source_rule_id"].startswith("HARD_BLOCK:")


def test_stack_off_run_uses_the_unadjusted_threshold(built_pipeline):
    """§7.6 / acceptance §10 item 8: the same pipeline, one flag flipped."""
    record = _sample_record()
    on = Engine().bind(built_pipeline).score(record, {"fraud_interdiction": {"_stack_enabled":
                                                                              {"adjustment_stack_enabled": True}}})
    off = Engine().bind(built_pipeline).score(record, {"fraud_interdiction": {"_stack_enabled":
                                                                               {"adjustment_stack_enabled": False}}})
    assert on["mule_scam_amount_multiplier"] == 0.7
    assert off["mule_scam_amount_multiplier"] == 1.0
    assert off["fired_on_overlay_ids"] == []
    assert set(on["fired_on_overlay_ids"]) <= set(on["fired_rule_ids"])


def test_batch_and_real_time_paths_agree(built_pipeline):
    """The equivalence requirement (§5.17): score() per record and run() as a batch must produce
    identical fired_rule_ids and action_code for the same events -- proven, not assumed, because
    they are literally the same `decider` object run two different ways."""
    records = [_sample_record() for _ in range(3)]
    records[1]["amount"] = 500.0
    records[1]["beneficiary_first_payment"] = False
    records[2]["channel_code"] = 1

    scored = [Engine().bind(built_pipeline).score(r, {}) for r in records]
    batch = backtest.run_backtest(built_pipeline, {}, records)

    for i, single in enumerate(scored):
        assert batch["action_code"][i] == single["action_code"]
        assert list(batch["fired_rule_ids"][i]) == single["fired_rule_ids"]
        assert list(batch["shadow_fired_rule_ids"][i]) == single["shadow_fired_rule_ids"]


def test_backtest_summary_reports_hit_rate_and_value_blocked(built_pipeline):
    records = [_sample_record() for _ in range(2)]
    result = backtest.run_backtest(built_pipeline, {}, records)
    summary = backtest.summarise(result)
    assert summary["events"] == 2
    assert 0.0 <= summary["hit_rate"] <= 1.0


def test_decider_build_cli_succeeds():
    """Verifies serving exactly as SERVE.md documents: the real CLI, not a Python shortcut."""
    import decider as _decider

    from conftest import _SHARED_CORE

    repo_root = Path(_decider.__file__).resolve().parents[1]  # .../<repo>/decider/__init__.py

    env = dict(os.environ)
    env["DECIDER_API__CODE_PATH"] = str(_HERE)
    env["DECIDER_API__PIPELINE"] = "pipeline:build"
    env["DECIDER_CONFIG__BASEPATH"] = str(_CONFIGS.parent)
    env["DECIDER_API__MODE"] = "interpreted"
    env["PYTHONPATH"] = str(_SHARED_CORE)
    result = subprocess.run(["uv", "run", "--project", str(repo_root), "decider", "build"],
                             cwd=str(_HERE), env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "built config version" in (result.stdout + result.stderr)
