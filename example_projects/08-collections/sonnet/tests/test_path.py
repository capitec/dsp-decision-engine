"""§5.5: monotonic escalation, resets, roll mid-sequence, re-entry after curing."""
from collections_treatment import path, vocab


def _defaults(**overrides):
    base = dict(
        arrears_bucket_code=3, matrix_treatment_intensity=2,
        previous_episode_open=True, previous_path_position=2, previous_bucket=3,
        previous_intensity_ceiling=3, previous_channel_attempt_count=2,
        times_cured_12m=0, qualifying_payment=False, promise_captured=False,
        cured=False, dispute_raised=False, arrangement_activated=False,
    )
    base.update(overrides)
    return base


def test_intensity_never_decreases_within_an_episode_without_a_reset():
    out = path.resolve_path_position(**_defaults(matrix_treatment_intensity=1))
    intensity_ceiling = out[3]
    assert intensity_ceiling >= 3  # floored at what was already delivered (previous_intensity_ceiling)


def test_qualifying_payment_resets_to_entry_position_and_drops_the_ceiling_by_one():
    out = path.resolve_path_position(**_defaults(qualifying_payment=True))
    position, episode_open, _delta, ceiling, attempts, reset_event, kind = out
    assert position == 1
    assert ceiling == 2  # previous_intensity_ceiling (3) - 1
    assert attempts == 0
    assert reset_event == vocab.RESET_QUALIFYING_PAYMENT
    assert kind == "reset"


def test_cure_closes_the_episode():
    out = path.resolve_path_position(**_defaults(cured=True))
    position, episode_open, _delta, ceiling, attempts, reset_event, kind = out
    assert episode_open is False
    assert reset_event == vocab.RESET_CURE
    assert kind == "reset"


def test_dispute_holds_position_and_counters_unchanged():
    out = path.resolve_path_position(**_defaults(dispute_raised=True))
    position, episode_open, _delta, ceiling, attempts, reset_event, kind = out
    assert position == 2  # previous_path_position, unchanged
    assert attempts == 2  # previous_channel_attempt_count, unchanged
    assert kind == "hold"


def test_roll_to_worse_bucket_carries_counters_and_floors_intensity_at_the_greater_value():
    out = path.resolve_path_position(**_defaults(
        arrears_bucket_code=5, previous_bucket=3, matrix_treatment_intensity=1,
        previous_intensity_ceiling=4, previous_channel_attempt_count=6,
    ))
    position, episode_open, _delta, ceiling, attempts, reset_event, kind = out
    assert position == 1  # entry position of the new, worse bucket
    assert ceiling == 4  # the greater of matrix_treatment_intensity (1) and previous ceiling (4)
    assert attempts == 6  # counters carry across the roll -- fatigue is a client property
    assert kind == "escalate"


def test_roll_to_better_bucket_via_partial_payment_does_not_de_escalate():
    out = path.resolve_path_position(**_defaults(arrears_bucket_code=2, previous_bucket=3))
    position, episode_open, _delta, ceiling, attempts, reset_event, kind = out
    assert ceiling >= 3  # never drops just because the bucket improved without a qualifying payment
    assert kind == "hold"


def test_fresh_episode_re_entry_position_depends_on_cure_history():
    never_cured = path.resolve_path_position(**_defaults(previous_episode_open=False, times_cured_12m=0))
    cured_once = path.resolve_path_position(**_defaults(previous_episode_open=False, times_cured_12m=1))
    chronic = path.resolve_path_position(**_defaults(previous_episode_open=False, times_cured_12m=3))
    assert never_cured[0] == 1
    assert cured_once[0] == 3
    assert chronic[0] == 1  # chronic re-ager path: agent contact from day 1


def test_intensity_floor_never_lets_the_matrix_recommend_below_the_ceiling():
    assert path.apply_intensity_floor(matrix_treatment_intensity=1, intensity_ceiling=4) == 4
    assert path.apply_intensity_floor(matrix_treatment_intensity=5, intensity_ceiling=2) == 5


def test_interval_and_cap_check_evaluates_every_cap_not_only_the_first_breach():
    must_escalate, blocked, detail = path.interval_and_cap_check(
        channel_attempt_count=5, permitted_retries=3, contacts_today=4, spoken_contacts_today=3,
        contacts_rolling_7d=11, contacts_rolling_30d=25, days_since_last_treatment=0, cooling_off_days=5,
    )
    for name in ("retries_exhausted", "cooling_off", "daily_cap", "spoken_daily_cap",
                 "rolling_7d_cap", "rolling_30d_cap"):
        assert name in detail
    assert must_escalate is True
    assert blocked is True
