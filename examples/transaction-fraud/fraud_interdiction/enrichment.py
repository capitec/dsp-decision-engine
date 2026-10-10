"""How much of the enrichment this decision could rely on: velocity freshness, the source bitset and the mode."""
from __future__ import annotations

from decikit import missing_as, param

from fraud_interdiction import vocab as v


def velocity_completeness_band(
    velocity_count_1min_state: int, velocity_count_10min_state: int, velocity_sum_amount_1h_state: int,
) -> int:
    """A `VelocityBand`: materially degraded when a short window is absent or two aggregates are."""
    absent = ((velocity_count_1min_state == v.ABSENT) + (velocity_count_10min_state == v.ABSENT)
              + (velocity_sum_amount_1h_state == v.ABSENT))
    if velocity_count_1min_state == v.ABSENT or velocity_count_10min_state == v.ABSENT or absent > 1:
        return v.VelocityBand.MATERIALLY_DEGRADED
    if velocity_count_1min_state + velocity_count_10min_state + velocity_sum_amount_1h_state > 0:
        return v.VelocityBand.PARTIALLY_DEGRADED
    return v.VelocityBand.COMPLETE


def enrichment_degradation_code(
    velocity_completeness_band: int,
    model_score_available: bool = missing_as(True), client_profile_degraded: bool = missing_as(False),
    device_session_degraded: bool = missing_as(False), counterparty_degraded: bool = missing_as(False),
    sanctions_list_stale: bool = missing_as(False), mule_watchlist_stale: bool = missing_as(False),
) -> int:
    """One bit per degraded source (`vocab.BIT_*`)."""
    code = 0
    if client_profile_degraded:
        code |= v.BIT_CLIENT_PROFILE
    if device_session_degraded:
        code |= v.BIT_DEVICE_SESSION
    if counterparty_degraded:
        code |= v.BIT_COUNTERPARTY
    if velocity_completeness_band != v.VelocityBand.COMPLETE:
        code |= v.BIT_VELOCITY
    if not model_score_available:
        code |= v.BIT_MODEL_SCORE
    if sanctions_list_stale:
        code |= v.BIT_SANCTIONS
    if mule_watchlist_stale:
        code |= v.BIT_MULE_WATCHLIST
    return code


def degraded_mode_code(
    enrichment_degradation_code: int, velocity_completeness_band: int,
    sanctions_list_stale: bool = missing_as(False), mule_watchlist_stale: bool = missing_as(False),
    restricted_from_sources: int = param(2, ge=1),
) -> int:
    """A `Mode`. Fail closed when a list that gates payments is stale; restricted once enough sources degrade."""
    if sanctions_list_stale or mule_watchlist_stale:
        return v.Mode.FAIL_CLOSED
    degraded, bits = 0, enrichment_degradation_code
    while bits:
        degraded += bits & 1
        bits >>= 1
    if degraded >= restricted_from_sources or velocity_completeness_band == v.VelocityBand.MATERIALLY_DEGRADED:
        return v.Mode.RESTRICTED
    return v.Mode.REDUCED if degraded else v.Mode.NORMAL


def late_arrival_flag(event_timestamp_age_seconds: float = missing_as(0.0), late_after_seconds: float = param(300.0)) -> bool:
    """An event that arrives late is still assessed, but flagged."""
    return event_timestamp_age_seconds > late_after_seconds
