"""5.3 — device and session signals, and the cross-event deltas.

"These cross-event deltas are the account-takeover family's core features and
are why non-payment events are in scope at all — they exist only because logins,
device registrations and number changes were themselves assessed and recorded."
(spec §5.3)

That sentence is an architectural instruction: this module reads the engine's
*own* decision record store for prior events on the same account. It is the one
enrichment source whose upstream is this system. Which means an outage here is
self-inflicted and correlated, and `completeness.py` treats it as such.
"""

from __future__ import annotations

from decider2 import Observed, module, observed, param
from decider2.types import Instant


def hours_since_device_first_seen(device_id: str, client_id: str) -> Observed[float]:
    """Hours since this device was first seen on this account."""
    pass


def hours_since_any_device_change(client_id: str, event_timestamp: Instant) -> Observed[float]:
    """Hours since *any* device change on the account. Derived from prior 311s."""
    pass


def hours_since_sim_change(client_id: str, event_timestamp: Instant) -> Observed[float]:
    """Derived from prior 312 events. Late-arriving by nature: a SIM change
    notified at 14:00 may have occurred at 11:20, and this uses the occurrence
    instant, not the notification instant."""
    pass


def device_reputation_band(device_id: str, event_timestamp: Instant) -> Observed[int]:
    """12 bands from the versioned device table. ABSENT for the 11% of
    card-present events that legitimately carry no device — and that is
    NOT_APPLICABLE, not ABSENT, because a card-present terminal has no device
    fingerprint by construction (spec §4.2: "Absent is a value, not a failure")."""
    pass


def device_blocklisted(device_id: str, event_timestamp: Instant) -> Observed[bool]:
    """Blocked-device membership as at the instant. Feeds a hard block."""
    pass


def automation_indicator_bits(device_id: str, session_id: str) -> Observed[int]:
    """Emulator, rooting and automation flags as a bitset."""
    pass


def ip_locus_distance_band(session_id: str, client_id: str) -> Observed[int]:
    """IP-to-account-locus distance, banded. Bands not kilometres: an analyst
    tunes a band cut-off in the device table, not a rule threshold."""
    pass


def behavioural_biometrics_band(session_id: str) -> Observed[int]:
    """Supplied for some channels only. NOT_APPLICABLE elsewhere."""
    pass


DeviceSession = module(
    hours_since_device_first_seen,
    hours_since_any_device_change,
    hours_since_sim_change,
    device_reputation_band,
    device_blocklisted,
    automation_indicator_bits,
    ip_locus_distance_band,
    behavioural_biometrics_band,
    name="device_session",
    contract="contracts/feature_vector.json#/device",
)
