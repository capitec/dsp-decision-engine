"""`core.reason_codes`: ranking, primary reason, registry version (spec 00 §6.18)."""
import pytest
from decider import Engine

from credit_core.reason_codes import ReasonCode, ReasonCodeRegistry


def _registry():
    return ReasonCodeRegistry("v1", [
        ReasonCode(1, severity_rank=3, description="minor", is_regulatory=False),
        ReasonCode(2, severity_rank=1, description="most severe", is_regulatory=True),
        ReasonCode(3, severity_rank=2, description="middle", is_regulatory=True),
    ])


def test_ranks_most_severe_first_and_picks_the_primary():
    ordered, primary = _registry().rank([1, 2, 3])
    assert ordered == [2, 3, 1]
    assert primary == 2


def test_empty_fired_set_has_no_primary():
    ordered, primary = _registry().rank([])
    assert ordered == []
    assert primary is None


def test_duplicates_collapse():
    ordered, _ = _registry().rank([1, 1, 2])
    assert ordered == [2, 1]


def test_an_unregistered_code_is_an_error_not_a_silent_drop():
    """Dropping an unrecognised code during a port is the exact failure 00 §6.18 exists to prevent."""
    with pytest.raises(LookupError, match="999"):
        _registry().rank([999])


def test_duplicate_codes_at_registry_construction_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ReasonCodeRegistry("v1", [
            ReasonCode(1, 1, "a", False),
            ReasonCode(1, 2, "b", False),
        ])


def test_resolve_step_emits_the_registry_version():
    step = _registry().resolve_step()
    exe = Engine().bind(step, mode="interpreted")
    out = exe.score({"decline_reason_codes": [3, 2]})
    assert out["decline_reason_codes"] == [2, 3]
    assert out["primary_reason_code"] == 2
    assert out["reason_registry_version"] == "v1"


def test_resolve_step_handles_no_fired_reasons():
    step = _registry().resolve_step()
    exe = Engine().bind(step, mode="interpreted")
    out = exe.score({"decline_reason_codes": []})
    assert out["decline_reason_codes"] == []
    assert out["primary_reason_code"] is None
