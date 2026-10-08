import pytest

from credit_core.reason_codes import ReasonCode, ReasonCodeRegistry

REGISTRY = ReasonCodeRegistry("v1", [
    ReasonCode(1, severity_rank=3, description="minor", is_regulatory=False),
    ReasonCode(2, severity_rank=1, description="most severe", is_regulatory=True),
    ReasonCode(3, severity_rank=2, description="middle", is_regulatory=True),
    ReasonCode(4, severity_rank=2, description="as severe as 3", is_regulatory=False),
])


def test_codes_are_ranked_most_severe_first_and_the_first_is_primary():
    assert REGISTRY.rank([1, 2, 3]) == ([2, 3, 1], 2)


def test_equally_severe_codes_are_ordered_by_code():
    assert REGISTRY.rank([4, 3]) == ([3, 4], 3)


def test_duplicates_collapse():
    assert REGISTRY.rank([1, 1, 2]) == ([2, 1], 2)


def test_no_fired_codes_means_no_primary():
    assert REGISTRY.rank([]) == ([], None)


def test_a_code_outside_the_registry_is_an_error_not_a_silent_drop():
    with pytest.raises(LookupError, match=r"\[999\]"):
        REGISTRY.rank([2, 999])


def test_duplicate_codes_in_a_registry_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ReasonCodeRegistry("v1", [ReasonCode(1, 1, "a", False), ReasonCode(1, 2, "b", False)])


def test_membership():
    assert 2 in REGISTRY and 999 not in REGISTRY


def test_the_step_ranks_in_place_and_records_the_primary_and_the_registry_version(score):
    out = score(REGISTRY.resolve_step(), {"decline_reason_codes": [3, 2]})
    assert (out["decline_reason_codes"], out["primary_reason_code"], out["reason_registry_version"]) == ([2, 3], 2, "v1")


def test_the_step_handles_no_reasons(score):
    out = score(REGISTRY.resolve_step(), {"decline_reason_codes": []})
    assert out["decline_reason_codes"] == [] and out["primary_reason_code"] is None
