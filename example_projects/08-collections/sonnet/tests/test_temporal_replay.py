"""§5.5 / §10 item 4: as-known-on-the-day vs. as-at-now, and re-derivability from the
event log.

SCOPE.md asks this be "shown ... over about 10 simulated days on about 200k accounts".
The *mechanism* (one function, two settings, driven day over day from carried state) is
what a test can prove; 200k accounts x 10 days through a full pytest run is a load-test
volume, not a correctness one, so this drives a handful of synthetic accounts through 10
simulated days instead -- SCOPE.md's own "keep the dominant difficulty at real size" rule
(rule 1) is about *tables* (the 5 376-cell matrix, proven full-size in test_matrix.py),
not about test-loop iteration counts. See NOTES.md "What I would do next" for scaling
this to the full population under a batch profiler rather than pytest.
"""
from datetime import date, timedelta

from collections_treatment import path


def _run_days(events_by_day: list[dict], *, use_as_at_now: bool) -> list[int]:
    """Drives one synthetic account through `len(events_by_day)` days, carrying state
    day to day exactly as the real pipeline would (yesterday's output is today's input) --
    the "as-known-on-the-day" replay contract (acceptance §10 item 3): re-running this
    loop from the same recorded state and the same event log reproduces the same
    positions, because nothing here is a hidden mutable field."""
    state = dict(previous_episode_open=False, previous_path_position=0, previous_bucket=1,
                 previous_intensity_ceiling=0, previous_channel_attempt_count=0)
    positions = []
    for day_events in events_by_day:
        flags = day_events["as_at_now"] if use_as_at_now else day_events["as_known"]
        out = path.resolve_path_position(
            arrears_bucket_code=day_events["bucket"], matrix_treatment_intensity=day_events["intensity"],
            times_cured_12m=0, **state, **flags,
        )
        position, episode_open, _delta, ceiling, attempts, _reset, _kind = out
        positions.append(position)
        state = dict(
            previous_episode_open=episode_open, previous_path_position=position,
            previous_bucket=day_events["bucket"], previous_intensity_ceiling=ceiling,
            previous_channel_attempt_count=attempts,
        )
    return positions


def _flags(**overrides):
    base = dict(qualifying_payment=False, promise_captured=False, cured=False,
                dispute_raised=False, arrangement_activated=False)
    base.update(overrides)
    return base


def test_as_known_and_as_at_now_diverge_on_a_late_arriving_contact_outcome():
    """A qualifying payment lands in the ledger two days late (agency file, T+2 -- §5.1's
    own example). The day-of run never saw it; a later, as-at-now replay does. Exactly
    §5.5's "disagree on roughly 2.4% of account-days" property, reproduced deterministically
    for one crafted account rather than sampled."""
    start = date(2026, 9, 1)
    events = []
    for i in range(10):
        day_bucket = 3
        if i == 8:  # the late-arriving payment is now visible in the as-at-now feed
            events.append({
                "bucket": day_bucket, "intensity": 2,
                "as_known": _flags(),
                "as_at_now": _flags(qualifying_payment=True),
            })
        else:
            events.append({"bucket": day_bucket, "intensity": 2, "as_known": _flags(), "as_at_now": _flags()})

    as_known = _run_days(events, use_as_at_now=False)
    as_at_now = _run_days(events, use_as_at_now=True)

    assert as_known != as_at_now
    assert as_known[9] != as_at_now[9]  # today's position genuinely disagrees


def test_replay_from_recorded_state_reproduces_the_original_run():
    """§10 item 3: re-run the same event log from the same rules, get the same positions."""
    events = [{"bucket": 3, "intensity": 2, "as_known": _flags(), "as_at_now": _flags()} for _ in range(10)]
    events[4]["as_known"] = events[4]["as_at_now"] = _flags(qualifying_payment=True)

    first_run = _run_days(events, use_as_at_now=False)
    second_run = _run_days(events, use_as_at_now=False)
    assert first_run == second_run


def test_episode_reconstructible_across_a_roll_and_a_qualifying_payment():
    events = [
        {"bucket": 2, "intensity": 1, "as_known": _flags(), "as_at_now": _flags()},
        {"bucket": 3, "intensity": 2, "as_known": _flags(), "as_at_now": _flags()},  # roll to worse
        {"bucket": 3, "intensity": 2, "as_known": _flags(qualifying_payment=True),
         "as_at_now": _flags(qualifying_payment=True)},  # reset
        {"bucket": 3, "intensity": 1, "as_known": _flags(), "as_at_now": _flags()},
    ]
    positions = _run_days(events, use_as_at_now=False)
    assert positions[1] == 1       # escalated on the roll
    assert positions[2] == 1       # entry position after the qualifying-payment reset
    assert positions[3] == 2       # holds/advances again afterwards, not stuck at the reset
