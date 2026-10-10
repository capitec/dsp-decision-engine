import pytest
from decikit import ConfigurableStep, flow
from decikit.lab import diff

from credit_core.reason_codes import Code, ReasonCodes


def codes(version="v1", **severity):
    return ReasonCodes(version=version, codes=[
        Code(code=1, severity_rank=3, description="minor", is_regulatory=False),
        Code(code=2, severity_rank=severity.get("two", 1), description="most severe", is_regulatory=True),
        Code(code=3, severity_rank=2, description="middle", is_regulatory=True),
        Code(code=4, severity_rank=2, description="as severe as 3", is_regulatory=False),
    ])


def same(a, b):
    return diff.structure(flow(a, name="t"), flow(b, name="t")).identical


def test_codes_are_ranked_most_severe_first_and_the_first_is_primary():
    assert codes().rank([1, 2, 3]) == ([2, 3, 1], 2)


def test_equally_severe_codes_are_ordered_by_code():
    assert codes().rank([4, 3]) == ([3, 4], 3)


def test_duplicates_collapse():
    assert codes().rank([1, 1, 2]) == ([2, 1], 2)


def test_no_fired_codes_means_no_primary():
    assert codes().rank([]) == ([], None)


def test_a_code_outside_the_set_is_an_error_not_a_silent_drop():
    with pytest.raises(LookupError, match=r"\[999\]"):
        codes().rank([2, 999])


def test_duplicate_codes_are_rejected_when_the_document_loads():
    twice = {"type": "reason_codes", "version": "v1", "codes": [
        {"code": 1, "severity_rank": 1, "description": "a", "is_regulatory": False},
        {"code": 1, "severity_rank": 2, "description": "b", "is_regulatory": False}]}
    with pytest.raises(ValueError, match="duplicate"):
        ConfigurableStep.load(twice)


def test_membership():
    assert 2 in codes() and 999 not in codes()


def test_the_step_ranks_in_place_and_records_the_primary_and_the_version(score):
    step = codes()
    assert step.name == "rank_reasons"
    out = score(step, {"decline_reason_codes": [3, 2]})
    assert (out["decline_reason_codes"], out["primary_reason_code"], out["reason_registry_version"]) == ([2, 3], 2, "v1")


def test_the_step_handles_no_reasons(score):
    out = score(codes(), {"decline_reason_codes": []})
    assert out["decline_reason_codes"] == [] and out["primary_reason_code"] is None


def test_the_step_raises_on_an_unknown_code(score):
    with pytest.raises(LookupError, match=r"\[999\]"):
        score(codes(), {"decline_reason_codes": [999]})


def test_the_same_codes_are_unchanged_in_the_diff():
    assert same(codes(), ConfigurableStep.load(codes().model_dump_json()))


def test_changing_a_severity_shows_in_the_diff():
    assert not same(codes(), codes(two=99))


def test_changing_the_version_shows_in_the_diff():
    assert not same(codes(), codes("v2"))


def test_rewording_a_description_is_not_behaviour():
    reworded = ReasonCodes(version="v1", codes=[c.model_copy(update={"description": "reworded"}) for c in codes().codes])
    assert same(codes(), reworded)
