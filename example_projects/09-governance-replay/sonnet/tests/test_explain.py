"""Single-decision explanation, three audiences (spec 09 §5.2)."""
import json

import pytest

from governance import evidence_store, explain, flows


@pytest.fixture(params=["01", "05"])  # these two have a decline/adverse-record reason on their sample
def evidence(request):
    adapter = flows.get(request.param)
    built = adapter.build("0.1.0")
    req = json.loads((adapter.project_dir() / "sample_request.json").read_text())
    return evidence_store.capture(adapter, built, req)


def test_consultant_rendering_is_short_and_names_the_dominant_reason(evidence):
    rendering = explain.explain(evidence, "consultant")
    assert rendering.text
    assert len(rendering.text.splitlines()) <= 5
    assert rendering.detail["primary_reason"]


def test_analyst_rendering_carries_every_reason_and_the_full_record(evidence):
    rendering = explain.explain(evidence, "analyst")
    assert rendering.detail["reason_registry_version"]
    assert rendering.detail["all_reasons"]
    assert rendering.detail["full_record"] == evidence.record
    assert rendering.detail["mechanism"]  # flow-specific detail (fired rules / cap chain / attribution)


def test_adjudicator_rendering_names_the_policy_version_and_every_reason(evidence):
    rendering = explain.explain(evidence, "adjudicator")
    assert rendering.detail["policy_version"] == evidence.config_version
    assert rendering.detail["reason_registry_version"]
    assert str(evidence.config_version) in rendering.text


def test_unknown_audience_is_rejected(evidence):
    with pytest.raises(ValueError):
        explain.explain(evidence, "regulator")


def test_adjusted_and_unadjusted_values_are_shown_side_by_side():
    """09 §5.14.5: explanation must show the overlay's adjusted and unadjusted values
    together -- 03's sample carries at least a score overlay pair."""
    adapter = flows.get("03")
    built = adapter.build("0.1.0")
    req = json.loads((adapter.project_dir() / "sample_request.json").read_text())
    evidence = evidence_store.capture(adapter, built, req)
    rendering = explain.explain(evidence, "analyst")
    pairs = rendering.detail["unadjusted_vs_adjusted"]
    assert "score" in pairs
    assert "adjusted" in pairs["score"] and "unadjusted" in pairs["score"]
