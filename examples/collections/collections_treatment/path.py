"""The escalation path over time: where the account is in its sequence, what it was already given, which caps bind.

Yesterday's position is an input, so it is stored and supplied by the caller. The path is worked out twice, once
from the events known on the day and once from the events as they are known now (late agency outcomes arrive
after the day), so audit can replay the first and analysis can use the second.
"""
from __future__ import annotations

from decider import allow_fallback, missing_as, param, step

from collections_treatment import vocab as v

PATH_FLAGS = ("qualifying_payment", "promise_captured", "cured", "dispute_raised", "arrangement_activated")
PATH_OUTPUTS = ("path_position", "episode_open", "intensity_ceiling", "channel_attempt_count", "reset_event",
                "escalation_or_hold")

# The position a fresh episode enters at, by cure history: never cured and chronic re-agers start at 1, an account
# cured once or twice re-defaults at 3 (automated-only positions skipped).
_ENTRY_POSITION = {0: 1, 1: 3, 2: 3}


def _reset_event(qualifying_payment, promise_captured, cured, dispute_raised, arrangement_activated) -> str:
    # A cure ends the episode whatever else happened that day; a payment outranks a promise.
    for flag, event in ((cured, v.RESET_CURE), (dispute_raised, v.RESET_DISPUTE),
                        (arrangement_activated, v.RESET_ARRANGEMENT_ACTIVATED),
                        (qualifying_payment, v.RESET_QUALIFYING_PAYMENT), (promise_captured, v.RESET_PROMISE_CAPTURED)):
        if flag:
            return event
    return ""


@allow_fallback
def resolve_path_position(
    arrears_bucket_code: int, matrix_treatment_intensity: int,
    previous_episode_open: bool = missing_as(False), previous_path_position: int = missing_as(0),
    previous_bucket: int = missing_as(0), previous_intensity_ceiling: int = missing_as(0),
    previous_channel_attempt_count: int = missing_as(0), times_cured_12m: float = missing_as(0.0),
    qualifying_payment: bool = missing_as(False), promise_captured: bool = missing_as(False),
    cured: bool = missing_as(False), dispute_raised: bool = missing_as(False),
    arrangement_activated: bool = missing_as(False),
) -> tuple[int, bool, int, int, str, str]:
    """Today's `path_position`, `episode_open`, `intensity_ceiling`, `channel_attempt_count`, `reset_event`
    (empty when none) and `escalation_or_hold` ("entry", "reset", "hold" or "escalate").

    Intensity never falls inside an episode without a reset. A roll to a worse bucket restarts the position but
    keeps the attempt count and the ceiling; a roll to a better one does not de-escalate.
    """
    reset = _reset_event(qualifying_payment, promise_captured, cured, dispute_raised, arrangement_activated)
    if not previous_episode_open:
        return _ENTRY_POSITION.get(int(times_cured_12m), 1), True, matrix_treatment_intensity, 0, "", "entry"
    if reset == v.RESET_CURE:
        return 0, False, 0, 0, reset, "reset"
    if reset in (v.RESET_DISPUTE, v.RESET_ARRANGEMENT_ACTIVATED, v.RESET_PROMISE_CAPTURED):
        return previous_path_position, True, previous_intensity_ceiling, previous_channel_attempt_count, reset, "hold"
    if reset == v.RESET_QUALIFYING_PAYMENT:
        return 1, True, max(1, previous_intensity_ceiling - 1), 0, reset, "reset"
    ceiling = max(matrix_treatment_intensity, previous_intensity_ceiling)
    if arrears_bucket_code > previous_bucket:
        return 1, True, ceiling, previous_channel_attempt_count, "", "escalate"
    return previous_path_position + 1, True, ceiling, previous_channel_attempt_count + 1, "", "hold"


resolve_path_position_step = step(resolve_path_position, outputs=PATH_OUTPUTS)

# The same function on the as-at-now flags, writing the `_as_at_now` columns.
resolve_path_position_as_at_now = resolve_path_position_step.relabel(
    reads={flag: f"{flag}_as_at_now" for flag in PATH_FLAGS},
    writes={name: f"{name}_as_at_now" for name in PATH_OUTPUTS},
).named("resolve_path_position_as_at_now")


def path_floored_intensity(matrix_treatment_intensity: int, intensity_ceiling: int) -> int:
    """The matrix intensity, raised to what has already been delivered this episode."""
    return max(matrix_treatment_intensity, intensity_ceiling)


@allow_fallback
def interval_and_cap_check(
    channel_attempt_count: int, matrix_permitted_retries: int, matrix_cooling_off_days: int,
    contacts_today: int = missing_as(0), spoken_contacts_today: int = missing_as(0),
    contacts_rolling_7d: int = missing_as(0), contacts_rolling_30d: int = missing_as(0),
    days_since_last_treatment: int = missing_as(999),
    daily_cap: int = param(3, ge=0), spoken_daily_cap: int = param(2, ge=0),
    rolling_7d_cap: int = param(10, ge=0), rolling_30d_cap: int = param(24, ge=0),
) -> tuple[bool, bool, str]:
    """`must_escalate` (retries used up), `interval_or_cap_blocked` (a cap or cooling-off binds today) and
    `interval_or_cap_detail`, which names every limit that was reached, not only the first."""
    breaches = [name for name, hit in (
        ("retries_exhausted", channel_attempt_count >= matrix_permitted_retries),
        ("cooling_off", days_since_last_treatment < matrix_cooling_off_days),
        ("daily_cap", contacts_today >= daily_cap),
        ("spoken_daily_cap", spoken_contacts_today >= spoken_daily_cap),
        ("rolling_7d_cap", contacts_rolling_7d >= rolling_7d_cap),
        ("rolling_30d_cap", contacts_rolling_30d >= rolling_30d_cap)) if hit]
    return "retries_exhausted" in breaches, any(b != "retries_exhausted" for b in breaches), ",".join(breaches)


interval_and_cap_check_step = step(
    interval_and_cap_check, outputs=("must_escalate", "interval_or_cap_blocked", "interval_or_cap_detail"))
