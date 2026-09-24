"""The escalation path over time (spec 08 §5.5) -- answers spec §13 Q1 and Q2.

**Q1 (a fourth core component kind?).** This slice's answer: no. Sequence
position is composed from (a) the matrix's own recommendation, (b) one
small, declared table of reset-event effects (`_RESET_EFFECTS` below --
data, not branches), and (c) a handful of monotonicity/carry rules that
are genuinely arithmetic, not lookups (§5.5's "asymmetric, must be stated
explicitly" roll rule, the intensity ceiling). Whether that composition
scales past the ~5 events and ~4 product families this slice covers before
someone copies the whole function is exactly the open question §13 Q1
poses; see NOTES.md "What I would do next".

**Q2 (temporal state between runs).** Yesterday's position is *read*, not
recomputed from 180M raw event rows (the request already carries the
window aggregates -- see `state.py`'s module docstring for why). What this
module derives, from that state, is the *decision*: whether today resets,
escalates or holds. That derivation is versioned logic (`_RESET_EFFECTS`
below is itself a declared artefact, addressable the same way a table
cell is), so replay means re-running this function against the recorded
prior-day state and this module's own version -- not re-deriving the prior
state from scratch every time. §10 item 3 (re-derivable from the event log)
is satisfied one level up: `tests/test_temporal_replay.py` drives this
function day over day from an event log and shows the position it produces
matches what was recorded.

**as-known-on-the-day vs as-at-now (§5.5, §10 item 4).** One parameter,
`late_data_included`, selects which of two input variants
(`contact_outcome_code` vs `contact_outcome_code_as_at_now`, both supplied
by the caller) this function reads -- not two implementations. See
`pipeline.py` for how both variants are run through the same steps.
"""
from __future__ import annotations

from decider import missing_as, step

from collections_treatment import vocab

# Reset event -> (path_position_effect, intensity_ceiling_effect, clears_counters).
# `path_position_effect` is one of "entry_of_current_bucket", "hold", "unchanged".
# A declared table, not an if/elif chain, so a sixth reset event is one more row.
_RESET_EFFECTS = {
    vocab.RESET_QUALIFYING_PAYMENT: ("entry_of_current_bucket", -1, True),
    vocab.RESET_PROMISE_CAPTURED: ("hold", 0, False),
    vocab.RESET_CURE: ("episode_closed", 0, True),
    vocab.RESET_DISPUTE: ("hold", 0, False),
    vocab.RESET_ARRANGEMENT_ACTIVATED: ("hold", 0, False),
}

# Prior-cure-history -> (entry_position, bucket_floor, chronic) -- §5.5 "Re-entry after curing".
_REENTRY = {
    0: (1, None, False),   # no cure in 12 months
    1: (3, None, False),   # cured once, re-defaulted within 90 days
    2: (3, 3, False),      # cured twice in 12 months
}
_CHRONIC_ENTRY = (1, None, True)  # cured 3+ times: chronic re-ager path


def _detect_reset_event(
    qualifying_payment: bool, promise_captured: bool, cured: bool,
    dispute_raised: bool, arrangement_activated: bool,
) -> str | None:
    # Evaluated in a declared priority order (cure is terminal; a payment that also
    # clears a promise is still "qualifying payment", since that is the stronger reset).
    if cured:
        return vocab.RESET_CURE
    if dispute_raised:
        return vocab.RESET_DISPUTE
    if arrangement_activated:
        return vocab.RESET_ARRANGEMENT_ACTIVATED
    if qualifying_payment:
        return vocab.RESET_QUALIFYING_PAYMENT
    if promise_captured:
        return vocab.RESET_PROMISE_CAPTURED
    return None


def _entry_position(times_cured_12m: int) -> tuple[int, int | None, bool]:
    if times_cured_12m >= 3:
        return _CHRONIC_ENTRY
    return _REENTRY.get(times_cured_12m, _REENTRY[0])


def resolve_path_position(
    arrears_bucket_code: int,
    matrix_treatment_intensity: int,
    previous_episode_open: bool = missing_as(False),
    previous_path_position: int = missing_as(0),
    previous_bucket: int = missing_as(0),
    previous_intensity_ceiling: int = missing_as(0),
    previous_channel_attempt_count: int = missing_as(0),
    times_cured_12m: float = missing_as(0.0),
    qualifying_payment: bool = missing_as(False),
    promise_captured: bool = missing_as(False),
    cured: bool = missing_as(False),
    dispute_raised: bool = missing_as(False),
    arrangement_activated: bool = missing_as(False),
) -> tuple[int, bool, int, int, int, str, str]:
    """One call for both the as-known and as-at-now variants -- the caller (`pipeline.py`)
    supplies which set of `qualifying_payment`/`cured`/... flags to use for each; the
    function itself has no notion of "now". Returns: `path_position`, `episode_open`,
    `episode_id_delta` (1 if a new episode opened this call, else 0 -- the caller
    increments a running counter), `intensity_ceiling`, `channel_attempt_count`,
    `reset_event` (empty string if none), `escalation_or_hold` ("escalate"|"hold"|
    "reset"|"entry")."""
    reset_event = _detect_reset_event(
        bool(qualifying_payment), bool(promise_captured), bool(cured),
        bool(dispute_raised), bool(arrangement_activated),
    )

    if not previous_episode_open:
        # A fresh episode: entry position depends on cure history (§5.5 re-entry table).
        # The bucket-floor half of that table is applied upstream, in `state.py`'s
        # `arrears_bucket_code` (the one place §4.3's and §5.5's two floor rules combine).
        entry, _floor, _chronic = _entry_position(times_cured_12m)
        return entry, True, 1, matrix_treatment_intensity, 0, "", "entry"

    if reset_event == vocab.RESET_CURE:
        return 0, False, 0, 0, 0, reset_event, "reset"

    if reset_event in (vocab.RESET_DISPUTE, vocab.RESET_ARRANGEMENT_ACTIVATED, vocab.RESET_PROMISE_CAPTURED):
        # "Sequence held" (§5.5 table): position and counters unchanged.
        return previous_path_position, True, 0, previous_intensity_ceiling, \
            previous_channel_attempt_count, reset_event, "hold"

    if reset_event == vocab.RESET_QUALIFYING_PAYMENT:
        # Entry position of the *current* bucket; counters clear; ceiling drops by one.
        new_ceiling = max(1, previous_intensity_ceiling - 1)
        return 1, True, 0, new_ceiling, 0, reset_event, "reset"

    # No reset: rolling mid-sequence vs. plain escalation/hold (§5.5 "Rolling mid-sequence").
    if arrears_bucket_code > previous_bucket:
        # Roll to a worse bucket: position moves to the new bucket's entry position;
        # counters carry (fatigue is a property of the client, not the bucket); the
        # ceiling is the greater of the new bucket's entry intensity and what was
        # already delivered -- the "intensity floor carries" rule (§5.5).
        ceiling = max(matrix_treatment_intensity, previous_intensity_ceiling)
        return 1, True, 0, ceiling, previous_channel_attempt_count, "", "escalate"

    # A roll to a *better* bucket via partial payment does not de-escalate on its own.
    ceiling = max(previous_intensity_ceiling, matrix_treatment_intensity)
    position = previous_path_position + 1
    return position, True, 0, ceiling, previous_channel_attempt_count + 1, "", "hold"


resolve_path_position_step = step(
    resolve_path_position,
    outputs=("path_position", "episode_open", "episode_opened_delta", "intensity_ceiling",
              "channel_attempt_count", "reset_event", "escalation_or_hold"),
)


def apply_intensity_floor(matrix_treatment_intensity: int, intensity_ceiling: int) -> int:
    """§5.5 "Escalation is monotonic within an episode: treatment intensity may not
    decrease except after a reset event" -- the matrix's recommendation is floored at
    whatever intensity has already been delivered this episode."""
    return max(matrix_treatment_intensity, intensity_ceiling)


apply_intensity_floor_step = step(apply_intensity_floor, output="path_floored_intensity")


def interval_and_cap_check(
    channel_attempt_count: int,
    permitted_retries: int = missing_as(4),
    contacts_today: int = missing_as(0),
    spoken_contacts_today: int = missing_as(0),
    contacts_rolling_7d: int = missing_as(0),
    contacts_rolling_30d: int = missing_as(0),
    days_since_last_treatment: int = missing_as(999),
    cooling_off_days: int = missing_as(0),
) -> tuple[bool, bool, str]:
    """§5.5 "minimum intervals and maximum attempts" plus the absolute per-client caps
    (§5.5): every check is evaluated (not short-circuited) and the binding one, if any,
    is named -- the same "individually attributable" discipline as suspensions."""
    breaches = []
    if channel_attempt_count >= permitted_retries:
        breaches.append("retries_exhausted")
    if days_since_last_treatment < cooling_off_days:
        breaches.append("cooling_off")
    if contacts_today >= 3:
        breaches.append("daily_cap")
    if spoken_contacts_today >= 2:
        breaches.append("spoken_daily_cap")
    if contacts_rolling_7d >= 10:
        breaches.append("rolling_7d_cap")
    if contacts_rolling_30d >= 24:
        breaches.append("rolling_30d_cap")
    must_escalate = "retries_exhausted" in breaches
    blocked_today = any(b in breaches for b in
                         ("cooling_off", "daily_cap", "spoken_daily_cap", "rolling_7d_cap", "rolling_30d_cap"))
    return must_escalate, blocked_today, ",".join(breaches)


interval_and_cap_check_step = step(
    interval_and_cap_check, outputs=("must_escalate", "interval_or_cap_blocked", "interval_or_cap_detail"),
)
