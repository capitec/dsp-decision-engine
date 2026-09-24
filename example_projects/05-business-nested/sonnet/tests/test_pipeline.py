"""Integration: `pipeline.build()` end to end, over `sample_request.json`.

Exercises the 09-C evidence minimum SCOPE.md requires of every slice, plus
this spec's own headline acceptance criteria (§10 items 1-3): the decline
attributes to (entity, event set), and both the outcome and the attribution
are invariant under shuffling the entity list and each entity's event list.
"""
import datetime
import json
import random
from pathlib import Path

import polars as pl
import pytest
from decider import Engine

import pipeline

ROOT = Path(__file__).resolve().parents[1]


def _typed(record: dict) -> dict:
    record = dict(record)
    record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
    record["entities"] = [dict(e) for e in record["entities"]]
    for entity in record["entities"]:
        entity["adverse_events"] = [dict(ev) for ev in entity["adverse_events"]]
        for event in entity["adverse_events"]:
            event["event_date"] = datetime.date.fromisoformat(event["event_date"])
    return record


@pytest.fixture(scope="module")
def engine():
    return Engine().bind(pipeline.build())


@pytest.fixture
def record():
    return _typed(json.loads((ROOT / "sample_request.json").read_text()))


def test_the_sample_request_scores_end_to_end(engine, record):
    out = engine.score(record, {})
    assert out["outcome_code"] in (1, 2, 3, 4)
    assert out["decision_id"] == record["decision_id"]


def test_decline_attributes_to_an_entity_and_a_non_empty_event_set(engine, record):
    """Acceptance §10 item 1: every decline attributable to an entity carries a
    non-null entity id and a non-empty attributing event set."""
    out = engine.score(record, {})
    assert out["outcome_code"] == 4  # sample_request.json is built to decline via entity 1's judgment
    assert out["attributing_entity_id"] is not None
    assert len(out["attributing_event_ids"]) > 0
    assert out["attributing_entity_id"] in list(out["entity_id"])


def test_outcome_and_attribution_are_invariant_under_shuffling(engine, record):
    """Acceptance §10 item 3, demonstrated (a smaller sample than the spec's 1000 x 20 --
    SCOPE.md's slice budget -- but the same property, over 10 shuffles)."""
    random.seed(7)
    baseline = engine.score(record, {})
    for _ in range(10):
        shuffled = _typed(json.loads((ROOT / "sample_request.json").read_text()))
        random.shuffle(shuffled["entities"])
        for entity in shuffled["entities"]:
            random.shuffle(entity["adverse_events"])
        out = engine.score(shuffled, {})
        assert out["outcome_code"] == baseline["outcome_code"]
        assert out["attributing_entity_id"] == baseline["attributing_entity_id"]
        assert sorted(out["attributing_event_ids"]) == sorted(baseline["attributing_event_ids"])
        assert list(out["entity_verdict_code"]) == list(baseline["entity_verdict_code"])


def test_batch_mode_agrees_with_record_at_a_time(engine, record):
    """Spec 05 §13 Q16: the same assessment run record-at-a-time and set-at-a-time."""
    single = engine.score(record, {})
    batch = engine.run(pl.DataFrame([record, record]), {})
    assert batch["outcome_code"].to_list() == [single["outcome_code"]] * 2
    assert batch["risk_grade"].to_list() == [single["risk_grade"]] * 2


def test_the_event_threshold_overlay_is_recorded_with_the_unadjusted_value(engine, record):
    """09 §5.15 item 7: an overlaid threshold, with the unoverlaid counterpart. Event 42
    (R7000, unsatisfied) is the one in `sample_request.json` built to sit between the
    overlaid material threshold (R5000, sector 412, critical) and the base one (R10000)
    -- the overlay must be the reason it classifies material rather than immaterial."""
    out = engine.score(record, {})
    idx = list(out["ev_event_id"]).index(42)
    assert out["ev_overlay_ids"][idx] == "ADJ-05-014"
    assert out["ev_threshold_base"][idx] == 10_000.0
    assert out["ev_threshold_used"][idx] == 5_000.0
    assert out["ev_severity_code"][idx] == 3  # MATERIAL, against the halved (overlaid) threshold
    assert out["ev_severity_code_unadjusted"][idx] == 2  # MINOR, against the base (unoverlaid) threshold


def test_replay_reproduces_the_decision_bit_for_bit(engine, record):
    """Acceptance §10 item 6, 09 §5.15 item 4 (no "today")."""
    first = engine.score(record, {})
    second = engine.score(record, {})
    assert first == second


def test_decider_build_stages_the_config_version(monkeypatch):
    """The BRIEF's own acceptance test: `decider build` must succeed unchanged."""
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
