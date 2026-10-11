import pytest

from fraud_interdiction import wording


@pytest.mark.parametrize("action,key", [(70, "freeze_account"), (60, "block_channel"), (50, "decline"),
                                         (40, "hold_for_review"), (30, "step_up")])
def test_an_action_the_client_sees_gets_generic_wording_for_that_action(decide, rule, sample, action, key):
    out = decide([rule("MS-0001", action_code=action)], sample)
    assert (out["client_visible"], out["client_wording_key"]) == (True, f"wording.{key}.v1")


@pytest.mark.parametrize("action", [10, 20])
def test_allow_and_monitor_are_invisible_to_the_client(decide, rule, sample, action):
    out = decide([rule("MS-0001", action_code=action)], sample)
    assert (out["client_visible"], out["client_wording_key"]) == (False, "wording.none.v1")


def test_the_wording_never_names_the_rule(decide, rule, sample):
    a = decide([rule("MS-0001", action_code=40)], sample)
    b = decide([rule("AA-0009", action_code=40, reason_code=5009)], sample)
    assert a["client_wording_key"] == b["client_wording_key"]


def test_a_channel_the_client_has_not_consented_to_suppresses_the_wording(decide, rule, sample):
    out = decide([rule("MS-0001", action_code=40)], sample, push_permitted=False)
    assert (out["client_visible"], out["client_wording_key"]) == (True, wording.SUPPRESSED)


def test_a_regulated_notice_goes_out_without_consent(decide, rule, sample):
    out = decide([rule("MS-0001", action_code=40)], sample, push_permitted=False, is_regulated_notice=True)
    assert out["client_wording_key"] == "wording.hold_for_review.v1"


def test_a_marketing_opt_out_suppresses_an_unregulated_notice(decide, rule, sample):
    out = decide([rule("MS-0001", action_code=40)], sample, marketing_opt_out=True)
    assert out["client_wording_key"] == wording.SUPPRESSED
