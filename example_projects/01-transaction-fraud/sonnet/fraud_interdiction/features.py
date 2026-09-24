"""Event admission, enrichment (stubbed) and the degraded-mode verdict (spec 01 §5.1-5.6).

SCOPE.md: "stub enrichment and the velocity store with the degradation
codes from §5.6." Enrichment values (counterparty, device, model score)
therefore arrive on the request already resolved -- as a real deployment's
enrichment stage would emit them -- rather than being looked up here from
1 024-row merchant tables etc. (out of scope for this slice; card-present
events, where those tables apply, are not in scope either -- SCOPE.md
restricts this slice to instant payments).

What *is* built at real depth: the degradation bitset and
`degraded_mode_code` verdict (§5.6), because a rule's stale/absent
behaviour and the suppressed/activated rule sets both depend on it, and
because 09 §5.15 item 8 requires degraded state to be recorded, not
inferred.
"""
from __future__ import annotations

from decider import missing_as, param, step

from fraud_interdiction import vocab

# --- Velocity aggregate freshness (§4.3, §5.4) -------------------------------
# Full scope is 168 aggregates (7 keys x 6 windows x 4 statistics); this slice
# tracks the three the spec calls out by name as the sharpest staleness
# examples -- client 1-minute count, client 10-minute count, client 1-hour
# amount sum -- rather than stubbing all 168 with no behavioural difference.
FRESH, STALE, ABSENT = 0, 1, 2
STALENESS_TOLERANCE_SECONDS = {"1min": 2, "10min": 10, "1h": 60, "24h": 60}


def velocity_completeness_band(
    velocity_count_1min_state: int, velocity_count_10min_state: int, velocity_sum_amount_1h_state: int,
) -> str:
    """complete | partially_degraded | materially_degraded (§5.4)."""
    states = (velocity_count_1min_state, velocity_count_10min_state, velocity_sum_amount_1h_state)
    absent_1min_or_10min = velocity_count_1min_state == ABSENT or velocity_count_10min_state == ABSENT
    absent_count = sum(1 for s in states if s == ABSENT)
    if absent_1min_or_10min or absent_count > 1:
        return "materially_degraded"
    if any(s == STALE for s in states):
        return "partially_degraded"
    return "complete"


velocity_completeness_band_step = step(velocity_completeness_band)


def enrichment_degradation_code(
    velocity_completeness_band: str,
    model_score_available: bool = missing_as(True),
    client_profile_degraded: bool = missing_as(False),
    device_session_degraded: bool = missing_as(False),
    counterparty_degraded: bool = missing_as(False),
    sanctions_list_stale: bool = missing_as(False),
    mule_watchlist_stale: bool = missing_as(False),
) -> int:
    """Bitset over the eight enrichment sources (§4.4, §5.6)."""
    code = 0
    if client_profile_degraded:
        code |= vocab.BIT_CLIENT_PROFILE
    if device_session_degraded:
        code |= vocab.BIT_DEVICE_SESSION
    if counterparty_degraded:
        code |= vocab.BIT_COUNTERPARTY
    if velocity_completeness_band != "complete":
        code |= vocab.BIT_VELOCITY
    if not model_score_available:
        code |= vocab.BIT_MODEL_SCORE
    if sanctions_list_stale:
        code |= vocab.BIT_SANCTIONS
    if mule_watchlist_stale:
        code |= vocab.BIT_MULE_WATCHLIST
    return code


enrichment_degradation_code_step = step(enrichment_degradation_code)


def degraded_mode_code(
    enrichment_degradation_code: int, velocity_completeness_band: str,
    sanctions_list_stale: bool = missing_as(False), mule_watchlist_stale: bool = missing_as(False),
) -> str:
    """normal | reduced | restricted | fail_closed (§5.6)."""
    if sanctions_list_stale or mule_watchlist_stale:
        return vocab.DEGRADED_MODE_FAIL_CLOSED
    degraded_sources = bin(enrichment_degradation_code).count("1")
    if degraded_sources >= 2 or velocity_completeness_band == "materially_degraded":
        return vocab.DEGRADED_MODE_RESTRICTED
    if degraded_sources == 1:
        return vocab.DEGRADED_MODE_REDUCED
    return vocab.DEGRADED_MODE_NORMAL


degraded_mode_code_step = step(degraded_mode_code)


# --- Event admission (§5.1) --------------------------------------------------

def event_timestamp_flag(event_timestamp_age_seconds: float = missing_as(0.0)) -> bool:
    """True when the event arrived more than 5 minutes late (§5.1) -- assessed regardless, but flagged."""
    return event_timestamp_age_seconds > 300.0


event_timestamp_flag_step = step(event_timestamp_flag, output="late_arrival_flag")
