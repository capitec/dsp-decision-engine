"""The pipeline as `decider serve` runs it: raw request bytes through the handler, in the default compiled mode."""
import asyncio
import json
from pathlib import Path

import polars as pl
import pytest
from decider.config import JsonFileStore
from decider.serving import RequestHandler

from fraud_interdiction.backtest import backtest, summarise

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "0.1.0"


@pytest.fixture(scope="module")
def handler():
    handler = RequestHandler(JsonFileStore(basepath=str(CONFIGS.parent)), "fraud_interdiction.pipeline:build",
                             sample_request=str(ROOT / "sample_request.json"))
    handler.stage()
    handler.activate()
    return handler


def post(handler, request: dict) -> dict:
    response = asyncio.run(handler.process_fn(json.dumps(request).encode(), "application/json", "application/json"))
    return json.loads(response.content)


@pytest.fixture
def raw_sample():
    return json.loads((ROOT / "sample_request.json").read_text())


def test_the_raw_sample_request_is_held_for_review_with_its_date_as_a_json_string(handler, raw_sample):
    assert isinstance(raw_sample["decision_date"], str)
    out = post(handler, raw_sample)
    assert (out["action_code"], out["action_source_rule_id"], out["primary_reason_code"]) == (40, "MS-0021", 3021)
    assert len(out["fired_rule_ids"]) == 20 and "MS-0208" in out["fired_rule_ids"]
    assert set(out["shadow_fired_rule_ids"]).isdisjoint(out["fired_rule_ids"]) and out["shadow_fired_rule_ids"]
    assert out["rule_set_version"] == "RS-2026.01"
    assert out["applied_adjustment_ids"] == ["ADJ-2026-002"] and out["mule_scam_amount_multiplier"] == 0.7
    assert (out["client_visible"], out["client_wording_key"]) == (True, "wording.hold_for_review.v1")


def test_an_amount_between_the_authored_and_the_overlay_threshold_is_held_only_because_of_the_overlay(handler, raw_sample):
    out = post(handler, {**raw_sample, "amount": 6000.0})  # MS-0208: R8 000 authored, R5 600 under the festive x0.7
    assert out["fired_on_overlay_ids"] == ["MS-0006", "MS-0018", "MS-0021", "MS-0208"]
    assert (out["action_code"], out["action_source_rule_id"]) == (40, "MS-0021")
    assert out["counterfactual_action_code"] == 30 and "MS-0208" not in out["counterfactual_fired_rule_ids"]


def test_a_small_payment_is_allowed_with_nothing_to_say(handler, raw_sample):
    out = post(handler, {**raw_sample, "amount": 300.0})
    assert (out["action_code"], out["fired_rule_ids"], out["decline_reason_codes"]) == (10, [], [])
    assert (out["client_visible"], out["primary_reason_code"]) == (False, None)


def test_a_sanctioned_payment_is_blocked_whatever_fires(handler, raw_sample):
    out = post(handler, {**raw_sample, "sanctions_list_match": True})
    assert out["action_code"] == 60 and out["action_source_rule_id"].startswith("HARD_BLOCK:sanctions")
    assert out["fired_rule_ids"]  # the rules were still evaluated and recorded
    assert out["hard_block_code"] == 1


def test_every_step_that_runs_in_python_says_so_on_purpose(handler):
    reasons = handler.module_fn().executable.fallbacks()
    assert reasons and all(r.startswith("@allow_fallback") for r in reasons.values()), reasons


def events(raw_sample) -> pl.DataFrame:
    from datetime import date

    rows = [{**raw_sample, "decision_date": date.fromisoformat(raw_sample["decision_date"]), **changes}
            for changes in ({}, {"amount": 300.0}, {"amount": 6000.0}, {"account_frozen": True, "amount": 50.0})]
    return pl.DataFrame(rows)


def test_a_batch_gives_the_answers_single_records_give(handler, raw_sample, params):
    executable = handler.module_fn().executable
    frame = events(raw_sample)
    batch = executable.run(frame, params)
    for i, record in enumerate(frame.to_dicts()):
        single = executable.score(record, params)
        assert batch["action_code"][i] == single["action_code"]
        assert batch["fired_rule_ids"][i].to_list() == single["fired_rule_ids"]
        assert batch["shadow_fired_rule_ids"][i].to_list() == single["shadow_fired_rule_ids"]


def test_a_backtest_costs_the_overlays_by_running_the_events_with_the_stack_off(handler, raw_sample, params):
    result = backtest(handler.module_fn().executable, events(raw_sample), params)
    assert result["with_overlays"]["events"] == result["without_overlays"]["events"] == 4
    assert result == {
        "with_overlays": {"events": 4, "blocked": 3, "hit_rate": 0.75, "value_blocked": 15250.0},
        "without_overlays": {"events": 4, "blocked": 2, "hit_rate": 0.5, "value_blocked": 9250.0}}


def test_the_summary_counts_holds_and_stops_and_adds_up_their_value():
    result = pl.DataFrame({"action_code": [10, 20, 30, 40, 50], "amount": [1.0, 2.0, 3.0, 100.005, 50.0]})
    assert summarise(result) == {"events": 5, "blocked": 2, "hit_rate": 0.4, "value_blocked": 150.0}
