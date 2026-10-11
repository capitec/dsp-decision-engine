import pytest

from credit_core.consent import channel_permitted, consent_verdict
from credit_core.vocab import ConsentChannel, ConsentVerdict


def test_a_marketing_opt_out_suppresses_a_marketing_message():
    assert consent_verdict(marketing_opt_out=True) == ConsentVerdict.SUPPRESSED


def test_a_regulated_notice_cannot_be_suppressed_by_a_marketing_preference():
    assert consent_verdict(marketing_opt_out=True, is_regulated_notice=True) == ConsentVerdict.REGULATED_NOTICE
    assert consent_verdict(data_sharing_consent=False, is_regulated_notice=True) == ConsentVerdict.REGULATED_NOTICE


def test_without_data_sharing_consent_a_message_is_suppressed():
    assert consent_verdict(data_sharing_consent=False) == ConsentVerdict.SUPPRESSED


def test_otherwise_a_message_is_permitted():
    assert consent_verdict() == ConsentVerdict.PERMITTED


@pytest.mark.parametrize("channel,flag", [
    (ConsentChannel.SMS, "sms_permitted"), (ConsentChannel.PUSH, "push_permitted"),
    (ConsentChannel.VOICE, "voice_permitted"), (ConsentChannel.EMAIL, "email_permitted")])
def test_a_permitted_message_goes_out_only_on_a_channel_the_customer_allows(channel, flag):
    assert channel_permitted(channel, ConsentVerdict.PERMITTED, **{flag: True}) is True
    assert channel_permitted(channel, ConsentVerdict.PERMITTED) is False


def test_other_channels_do_not_stand_in_for_the_one_used():
    assert channel_permitted(ConsentChannel.SMS, ConsentVerdict.PERMITTED, email_permitted=True, push_permitted=True) is False


def test_a_suppressed_message_goes_out_on_no_channel():
    assert channel_permitted(ConsentChannel.SMS, ConsentVerdict.SUPPRESSED, sms_permitted=True) is False


def test_a_regulated_notice_goes_out_on_any_channel():
    assert channel_permitted(ConsentChannel.VOICE, ConsentVerdict.REGULATED_NOTICE) is True


def test_the_verdict_step_returns_the_numeric_code(score):
    from credit_core.consent import consent_verdict_step

    assert score(consent_verdict_step, {"marketing_opt_out": True})["consent_verdict"] == 2
