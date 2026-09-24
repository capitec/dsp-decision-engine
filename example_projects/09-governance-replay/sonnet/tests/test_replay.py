"""Exact replay (spec 09 §5.1) over real evidence from all three flows this harness
consumes -- 01 (real-time rules), 03 (solve), 05 (nested)."""
import copy
import json

import pytest

from governance import evidence_store, flows, replay


@pytest.fixture(params=["01", "03", "05"])
def evidence(request):
    adapter = flows.get(request.param)
    built = adapter.build("0.1.0")
    req = json.loads((adapter.project_dir() / "sample_request.json").read_text())
    return evidence_store.capture(adapter, built, req)


def test_a_freshly_captured_decision_replays_bit_identically(evidence):
    verdict = replay.replay(evidence)
    assert verdict.verdict == "reproduced", verdict.first_divergence
    assert verdict.divergences == ()


def test_replay_by_id_reads_saved_evidence_from_the_store(evidence):
    evidence_store.save(evidence)
    verdict = replay.replay_by_id(evidence.flow_code, evidence.decision_id)
    assert verdict.verdict == "reproduced"


def test_a_tampered_record_does_not_reproduce(evidence):
    """A decision record that was edited after capture (09 §9 item 3's exact failure
    mode this harness exists to catch) must not silently replay clean."""
    tampered = copy.deepcopy(evidence)
    outcome_field = flows.get(evidence.flow_code).outcome_field
    object.__setattr__(tampered, "record", {**tampered.record, outcome_field: 999999})
    verdict = replay.replay(tampered)
    assert verdict.verdict == "not_reproduced"
    assert any(d.field == outcome_field for d in verdict.divergences)


def test_a_different_request_input_is_flagged_not_reproduced():
    """A request that was captured with a different input than what is now recorded
    (a corrupted "as received" input, 09 §5.15 item 6) must not reproduce."""
    adapter = flows.get("03")
    built = adapter.build("0.1.0")
    req = json.loads((adapter.project_dir() / "sample_request.json").read_text())
    evidence = evidence_store.capture(adapter, built, req)
    tampered_request = {**evidence.request, "requested_amount": 1.0}
    tampered = copy.deepcopy(evidence)
    object.__setattr__(tampered, "request", tampered_request)
    verdict = replay.replay(tampered)
    assert verdict.verdict == "not_reproduced"
