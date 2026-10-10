import pytest
from decikit import FusedEngine, InterpretedEngine, flow

from fraud_interdiction import enrichment, vocab
from fraud_interdiction.vocab import Mode, VelocityBand


@pytest.fixture(scope="module")
def assess():
    steps = flow(enrichment.velocity_completeness_band, enrichment.enrichment_degradation_code,
                 enrichment.degraded_mode_code, enrichment.late_arrival_flag, name="enrichment").emit(
        "velocity_completeness_band", "enrichment_degradation_code")
    executable = FusedEngine(strict_compile=True).bind(steps)  # raises if any of it leaves the kernel
    fresh = {"velocity_count_1min_state": 0, "velocity_count_10min_state": 0, "velocity_sum_amount_1h_state": 0}
    return lambda **record: executable.run({**fresh, **record})


def test_fresh_enrichment_is_normal(assess):
    out = assess()
    assert (out["velocity_completeness_band"], out["enrichment_degradation_code"]) == (VelocityBand.COMPLETE, 0)
    assert out["degraded_mode_code"] == Mode.NORMAL


def test_one_stale_aggregate_is_partly_degraded_velocity_and_reduces_the_mode(assess):
    out = assess(velocity_sum_amount_1h_state=vocab.STALE)
    assert out["velocity_completeness_band"] == VelocityBand.PARTIALLY_DEGRADED
    assert out["enrichment_degradation_code"] == vocab.BIT_VELOCITY
    assert out["degraded_mode_code"] == Mode.REDUCED


def test_an_absent_short_window_is_materially_degraded_and_restricts_the_mode(assess):
    out = assess(velocity_count_10min_state=vocab.ABSENT)
    assert out["velocity_completeness_band"] == VelocityBand.MATERIALLY_DEGRADED
    assert out["degraded_mode_code"] == Mode.RESTRICTED


def test_two_absent_aggregates_are_materially_degraded(assess):
    out = assess(velocity_count_1min_state=vocab.STALE, velocity_count_10min_state=vocab.STALE,
                 velocity_sum_amount_1h_state=vocab.ABSENT)
    assert out["velocity_completeness_band"] == VelocityBand.PARTIALLY_DEGRADED  # one absent, and not a short window
    both = assess(velocity_count_10min_state=vocab.STALE, velocity_sum_amount_1h_state=vocab.ABSENT,
                  velocity_count_1min_state=vocab.ABSENT)
    assert both["velocity_completeness_band"] == VelocityBand.MATERIALLY_DEGRADED


def test_each_degraded_source_sets_its_own_bit(assess):
    out = assess(client_profile_degraded=True, counterparty_degraded=True, model_score_available=False)
    assert out["enrichment_degradation_code"] == vocab.BIT_CLIENT_PROFILE | vocab.BIT_COUNTERPARTY | vocab.BIT_MODEL_SCORE
    assert out["degraded_mode_code"] == Mode.RESTRICTED  # three sources


def test_a_missing_model_score_flag_does_not_count_as_degraded(assess):
    assert assess()["enrichment_degradation_code"] == 0


def test_one_degraded_source_is_reduced_and_two_are_restricted(assess):
    assert assess(device_session_degraded=True)["degraded_mode_code"] == Mode.REDUCED
    assert assess(device_session_degraded=True, client_profile_degraded=True)["degraded_mode_code"] == Mode.RESTRICTED


@pytest.mark.parametrize("stale", ["sanctions_list_stale", "mule_watchlist_stale"])
def test_a_stale_gating_list_fails_closed(assess, stale):
    out = assess(**{stale: True})
    assert out["degraded_mode_code"] == Mode.FAIL_CLOSED
    assert out["enrichment_degradation_code"] == (vocab.BIT_SANCTIONS if stale.startswith("sanc") else vocab.BIT_MULE_WATCHLIST)


def test_restriction_threshold_is_a_param(assess):
    executable = InterpretedEngine().bind(flow(enrichment.degraded_mode_code, name="m"))
    two_sources = {"enrichment_degradation_code": 3, "velocity_completeness_band": 0}
    assert executable.run(two_sources)["degraded_mode_code"] == Mode.RESTRICTED
    assert executable.run(two_sources, {"m": {"degraded_mode_code": {"restricted_from_sources": 3}}})[
        "degraded_mode_code"] == Mode.REDUCED


def test_an_event_older_than_five_minutes_is_flagged_late(assess):
    assert assess(event_timestamp_age_seconds=301.0)["late_arrival_flag"] is True
    assert assess(event_timestamp_age_seconds=300.0)["late_arrival_flag"] is False
    assert assess()["late_arrival_flag"] is False
