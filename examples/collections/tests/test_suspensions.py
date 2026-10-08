from datetime import date

import pytest

from collections_treatment import suspensions, vocab as v
from collections_treatment.suspensions import add_business_days, evaluate_suspensions

DAY = date(2026, 9, 24)
AGENT_CALLS = {v.AGENT_CALL_LOW, v.AGENT_CALL_STANDARD, v.AGENT_CALL_HIGH}


def permitted(**flags) -> set[int]:
    return set(evaluate_suspensions(DAY, **flags)[4])


def test_every_suspension_in_force_is_named_not_the_first_found():
    codes, scopes, sources, expiries, _ = evaluate_suspensions(
        DAY, debt_review_stage_code=1, debt_review_case_number="DR-9001", outside_permitted_hours=True,
        consent_withdrawn_channels=[1])
    assert codes == [101, 115, 116]
    assert len(scopes) == len(sources) == len(expiries) == 3
    assert sources[0] == "debt_review_registry:DR-9001"
    assert scopes[2] == "channel:1"


def test_no_suspension_leaves_every_treatment_permitted():
    assert permitted() == set(v.TREATMENT_NAMES)


@pytest.mark.parametrize("flags", [
    {"debt_review_stage_code": 1}, {"debt_review_stage_code": 2}, {"debt_review_stage_code": 3},
    {"hardship_arrangement_performing": True}, {"complaint_open": True}, {"ombud_referral_open": True},
    {"administration_order_active": True}, {"deceased": True}, {"insolvency_active": True},
    {"prescription_date": date(2026, 9, 1)}, {"debt_review_stage_code": 4},
])
def test_a_suspension_that_forbids_contact_leaves_no_agent_call_permitted(flags):
    assert not AGENT_CALLS & permitted(**flags)


def test_debt_review_application_still_permits_a_statutory_notice_and_nothing_else():
    assert permitted(debt_review_stage_code=1) == {v.NO_ACTION, v.PRE_LEGAL_NOTICE, v.WRITE_OFF_RECOMMENDATION}


def test_deceased_blocks_every_client_action_but_not_the_internal_write_off_recommendation():
    assert permitted(deceased=True) == {v.NO_ACTION, v.WRITE_OFF_RECOMMENDATION}


def test_litigation_leaves_only_the_legal_handover():
    assert permitted(litigation_in_progress=True) == {v.NO_ACTION, v.LEGAL_HANDOVER, v.WRITE_OFF_RECOMMENDATION}


def test_a_dispute_blocks_agency_legal_and_calls_but_allows_written_contact():
    allowed = permitted(dispute_raised=True)
    assert {v.AUTOMATED_SMS, v.AUTOMATED_EMAIL, v.IN_APP_MESSAGE} <= allowed
    assert not allowed & ({v.AGENCY_HANDOVER, v.PRE_LEGAL_NOTICE, v.LEGAL_HANDOVER} | AGENT_CALLS)


def test_withdrawn_consent_blocks_only_the_channel_it_covers():
    allowed = permitted(consent_withdrawn_channels=[3])  # voice
    assert not allowed & (AGENT_CALLS | {v.AUTOMATED_VOICE_MESSAGE})
    assert {v.AUTOMATED_SMS, v.AUTOMATED_EMAIL, v.FIELD_VISIT} <= allowed


def test_two_channels_withdrawn_are_two_suspensions():
    codes, scopes, *_ = evaluate_suspensions(DAY, consent_withdrawn_channels=[1, 4])
    assert codes == [116, 116] and scopes == ["channel:1", "channel:4"]


def test_the_notice_period_is_ten_business_days_from_delivery():
    delivered = date(2026, 8, 14)
    assert add_business_days(delivered, 10) == date(2026, 8, 28)
    assert 112 in evaluate_suspensions(date(2026, 8, 27), notice_delivered_date=delivered)[0]
    assert 112 not in evaluate_suspensions(date(2026, 8, 28), notice_delivered_date=delivered)[0]
    assert evaluate_suspensions(date(2026, 8, 27), notice_delivered_date=delivered)[3] == ["2026-08-28"]


def test_the_notice_period_blocks_only_the_legal_handover():
    assert permitted(notice_delivered_date=date(2026, 9, 18)) == set(v.TREATMENT_NAMES) - {v.LEGAL_HANDOVER}


def test_a_complaint_lifts_fifteen_business_days_after_it_is_logged():
    expiry = evaluate_suspensions(DAY, complaint_open=True, complaint_logged_date=date(2026, 9, 17))[3]
    assert expiry == ["2026-10-08"]


def test_a_complaint_logged_without_a_date_has_an_unknown_expiry_not_a_missing_one():
    assert evaluate_suspensions(DAY, complaint_open=True)[3] == ["unknown"]


def test_a_promise_in_force_lets_only_the_sms_reminder_through():
    codes, *_, allowed = evaluate_suspensions(DAY, promise_date=date(2026, 9, 23))
    assert codes == [119]
    assert {v.AUTOMATED_SMS} <= set(allowed) and not {v.AUTOMATED_EMAIL, *AGENT_CALLS} & set(allowed)
    assert 119 not in evaluate_suspensions(DAY, promise_date=date(2026, 9, 18))[0]


def test_debt_review_in_default_protects_the_client_until_sixty_days_after_the_default():
    inside = evaluate_suspensions(date(2026, 9, 1), debt_review_stage_code=4, debt_review_default_date=date(2026, 8, 1))
    outside = evaluate_suspensions(date(2026, 9, 30), debt_review_stage_code=4, debt_review_default_date=date(2026, 8, 1))
    assert inside[0] == [104] and inside[3] == ["2026-09-30"]
    assert outside[0] == []


def test_prescription_is_a_suspension_only_once_the_date_is_reached():
    assert evaluate_suspensions(DAY, prescription_date=date(2026, 9, 24))[0] == [114]
    assert evaluate_suspensions(DAY, prescription_date=date(2026, 9, 25))[0] == []


def test_pre_prescription_flag_covers_the_sixty_days_before_the_date():
    assert suspensions.pre_prescription_flag(DAY, date(2026, 11, 20)) is True
    assert suspensions.pre_prescription_flag(DAY, date(2026, 11, 24)) is False
    assert suspensions.pre_prescription_flag(DAY, date(2026, 9, 1)) is False
    assert suspensions.pre_prescription_flag(DAY, None) is False


# The whole pipeline: the matrix recommends an agent call and a suspension must stop it.
AGENT_CALL_ACCOUNT = {"days_past_due": 70, "days_since_last_treatment": 999}


def test_without_a_suspension_the_account_is_called(decide):
    out = decide(**AGENT_CALL_ACCOUNT)
    assert out["treatment_code"] in AGENT_CALLS and out["allocated"] is True


@pytest.mark.parametrize("changes", [
    {"debt_review_stage_code": 1, "debt_review_case_number": "DR-9001"},
    {"hardship_arrangement_performing": True},
])
def test_a_suspended_account_gets_no_action_and_is_never_allocated(decide, changes):
    out = decide(**AGENT_CALL_ACCOUNT, **changes)
    assert out["overlaid_treatment_code"] in AGENT_CALLS
    assert (out["treatment_code"], out["suspension_blocked"]) == (v.NO_ACTION, True)
    assert out["allocated"] is False and out["non_selection_reason_code"] == v.NS_SUSPENDED
    assert out["pool_code"] == v.NO_POOL


def test_a_blocked_treatment_is_not_downgraded_to_a_cheaper_one(decide):
    out = decide(**AGENT_CALL_ACCOUNT, consent_withdrawn_channels=[3])  # voice consent withdrawn
    assert out["treatment_code"] == v.NO_ACTION and out["suspension_codes"] == [116]


def test_a_suspension_that_does_not_cover_the_treatment_leaves_it_alone(decide):
    out = decide(**AGENT_CALL_ACCOUNT, consent_withdrawn_channels=[4])  # email only
    assert out["treatment_code"] in AGENT_CALLS and out["suspension_blocked"] is False


def test_a_promise_in_force_is_reported_as_such(decide):
    out = decide(**AGENT_CALL_ACCOUNT, consent_withdrawn_channels=[], promise_date="2026-09-23")
    assert out["non_selection_reason_code"] == v.NS_PROMISE_IN_FORCE


def test_the_decision_names_the_permitted_set_and_the_suspensions_behind_it(decide):
    out = decide(**AGENT_CALL_ACCOUNT, consent_withdrawn_channels=[], debt_review_stage_code=1)
    assert out["suspension_codes"] == [101] and out["suspension_expiries"] == ["event:stage_change"]
    assert out["permitted_treatment_codes"] == [v.NO_ACTION, v.PRE_LEGAL_NOTICE, v.WRITE_OFF_RECOMMENDATION]
