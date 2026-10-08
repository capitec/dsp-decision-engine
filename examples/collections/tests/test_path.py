import pytest

from collections_treatment import path, vocab as v


def resolve(**changes):
    args = dict(arrears_bucket_code=3, matrix_treatment_intensity=2, previous_episode_open=True, previous_path_position=2,
                previous_bucket=3, previous_intensity_ceiling=3, previous_channel_attempt_count=2)
    return dict(zip(path.PATH_OUTPUTS, path.resolve_path_position(**args | changes)))


def test_intensity_never_falls_inside_an_episode_without_a_reset():
    assert resolve(matrix_treatment_intensity=1)["intensity_ceiling"] == 3


def test_a_qualifying_payment_returns_to_the_entry_position_and_lowers_the_ceiling_by_one():
    out = resolve(qualifying_payment=True)
    assert (out["path_position"], out["intensity_ceiling"], out["channel_attempt_count"]) == (1, 2, 0)
    assert (out["reset_event"], out["escalation_or_hold"]) == (v.RESET_QUALIFYING_PAYMENT, "reset")


def test_a_cure_closes_the_episode():
    out = resolve(cured=True)
    assert (out["episode_open"], out["path_position"], out["reset_event"]) == (False, 0, v.RESET_CURE)


@pytest.mark.parametrize("flag, event", [("dispute_raised", v.RESET_DISPUTE), ("promise_captured", v.RESET_PROMISE_CAPTURED),
                                         ("arrangement_activated", v.RESET_ARRANGEMENT_ACTIVATED)])
def test_a_dispute_promise_or_arrangement_holds_position_and_counters(flag, event):
    out = resolve(**{flag: True})
    assert (out["path_position"], out["channel_attempt_count"], out["intensity_ceiling"]) == (2, 2, 3)
    assert (out["reset_event"], out["escalation_or_hold"]) == (event, "hold")


def test_a_cure_outranks_a_payment_on_the_same_day():
    assert resolve(cured=True, qualifying_payment=True)["reset_event"] == v.RESET_CURE


def test_a_roll_to_a_worse_bucket_restarts_the_position_but_carries_the_counters_and_the_ceiling():
    out = resolve(arrears_bucket_code=5, previous_bucket=3, matrix_treatment_intensity=1, previous_intensity_ceiling=4,
                  previous_channel_attempt_count=6)
    assert (out["path_position"], out["intensity_ceiling"], out["channel_attempt_count"]) == (1, 4, 6)
    assert out["escalation_or_hold"] == "escalate"


def test_a_roll_to_a_worse_bucket_raises_the_ceiling_to_the_new_matrix_intensity():
    assert resolve(arrears_bucket_code=5, previous_bucket=3, matrix_treatment_intensity=5)["intensity_ceiling"] == 5


def test_a_roll_to_a_better_bucket_does_not_de_escalate():
    out = resolve(arrears_bucket_code=2, previous_bucket=3)
    assert out["intensity_ceiling"] == 3 and out["escalation_or_hold"] == "hold"
    assert (out["path_position"], out["channel_attempt_count"]) == (3, 3)


@pytest.mark.parametrize("cures, position", [(0, 1), (1, 3), (2, 3), (3, 1)])
def test_a_new_episode_enters_by_cure_history_and_a_chronic_re_ager_starts_at_one(cures, position):
    out = resolve(previous_episode_open=False, times_cured_12m=cures)
    assert (out["path_position"], out["escalation_or_hold"]) == (position, "entry")


def test_the_intensity_floor_never_lets_the_matrix_recommend_below_the_ceiling():
    assert path.path_floored_intensity(1, 4) == 4
    assert path.path_floored_intensity(5, 2) == 5


CAPS = dict(channel_attempt_count=5, matrix_permitted_retries=3, matrix_cooling_off_days=5, contacts_today=4,
            spoken_contacts_today=3, contacts_rolling_7d=11, contacts_rolling_30d=25, days_since_last_treatment=0)


def test_every_limit_that_binds_is_named_not_only_the_first():
    must_escalate, blocked, detail = path.interval_and_cap_check(**CAPS)
    assert detail.split(",") == ["retries_exhausted", "cooling_off", "daily_cap", "spoken_daily_cap", "rolling_7d_cap",
                                 "rolling_30d_cap"]
    assert (must_escalate, blocked) == (True, True)


def test_used_up_retries_escalate_without_blocking_today():
    assert path.interval_and_cap_check(channel_attempt_count=3, matrix_permitted_retries=3, matrix_cooling_off_days=0) \
        == (True, False, "retries_exhausted")


def test_nothing_binds_for_a_fresh_account():
    assert path.interval_and_cap_check(channel_attempt_count=0, matrix_permitted_retries=3, matrix_cooling_off_days=2) \
        == (False, False, "")


def test_the_caps_are_params_so_compliance_can_change_them_with_the_params_document(decide, params):
    tight = {**params, "collections_treatment": {**params["collections_treatment"],
                                                 "interval_and_cap_check": {**params["collections_treatment"]["interval_and_cap_check"],
                                                                            "rolling_7d_cap": 2}}}
    assert decide()["interval_or_cap_detail"] == ""
    out = decide(params_document=tight)  # the sample has 2 contacts in 7 days
    assert "rolling_7d_cap" in out["interval_or_cap_detail"] and out["non_selection_reason_code"] == v.NS_INTERVAL_OR_CAP
