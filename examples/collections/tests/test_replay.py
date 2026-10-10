"""The path is replayable: the same events and rules give the same positions, and a late event changes only the
as-at-now run."""
import pytest

from collections_treatment import path, vocab as v

NO_EVENTS = dict.fromkeys(path.PATH_FLAGS, False)


def run_days(days: list[dict], *, as_at_now: bool) -> list[int]:
    """One account through the days, each day's output carried in as the next day's `previous_*`."""
    carried = dict(previous_episode_open=False, previous_path_position=0, previous_bucket=1,
                   previous_intensity_ceiling=0, previous_channel_attempt_count=0)
    positions = []
    for day in days:
        events = NO_EVENTS | day.get("as_at_now" if as_at_now else "as_known", {})
        position, open_, ceiling, attempts, *_ = path.resolve_path_position(
            arrears_bucket_code=day["bucket"], matrix_treatment_intensity=day["intensity"], **carried, **events)
        positions.append(position)
        carried = dict(previous_episode_open=open_, previous_path_position=position, previous_bucket=day["bucket"],
                       previous_intensity_ceiling=ceiling, previous_channel_attempt_count=attempts)
    return positions


def quiet_days(n):
    return [{"bucket": 3, "intensity": 2} for _ in range(n)]


def test_replaying_the_same_events_gives_the_same_positions():
    days = quiet_days(10)
    days[4]["as_known"] = {"qualifying_payment": True}
    assert run_days(days, as_at_now=False) == run_days(days, as_at_now=False)
    assert run_days(days, as_at_now=False) == [1, 2, 3, 4, 1, 2, 3, 4, 5, 6]


def test_a_late_payment_changes_the_as_at_now_position_but_not_the_one_computed_on_the_day():
    days = quiet_days(10)
    days[8]["as_at_now"] = {"qualifying_payment": True}  # the agency file lands two days late
    as_known, as_at_now = run_days(days, as_at_now=False), run_days(days, as_at_now=True)
    assert as_known[:8] == as_at_now[:8]
    assert (as_known[9], as_at_now[9]) == (10, 2)


def test_an_episode_survives_a_roll_and_restarts_after_a_payment():
    days = [{"bucket": 2, "intensity": 1}, {"bucket": 3, "intensity": 2},
            {"bucket": 3, "intensity": 2, "as_known": {"qualifying_payment": True}}, {"bucket": 3, "intensity": 1}]
    assert run_days(days, as_at_now=False) == [1, 1, 1, 2]


def test_the_pipeline_runs_the_path_twice_and_keeps_both(decide):
    # The sample rolled from bucket 2 to 3, but a late arrangement activation is already known now.
    out = decide()
    assert (out["escalation_or_hold"], out["path_position"]) == ("escalate", 1)
    assert (out["escalation_or_hold_as_at_now"], out["path_position_as_at_now"]) == ("hold", 2)
    assert out["reset_event_as_at_now"] == v.RESET_ARRANGEMENT_ACTIVATED and out["reset_event"] == ""


def test_the_as_known_run_never_reads_the_as_at_now_events(decide):
    late = decide(cured_as_at_now=True)
    assert late["episode_open_as_at_now"] is False and late["episode_open"] is True
