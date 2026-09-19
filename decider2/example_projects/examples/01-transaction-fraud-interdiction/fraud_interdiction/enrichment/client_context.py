"""5.2 — client and account context, as at `event_timestamp`.

"Every attribute value used, and the snapshot it came from. Not a pointer to the
profile store — the value. The store is mutable and will not hold this value in
540 days." (spec §5.2)

So every output here is a *recorded* column of the feature vector. Nothing in
the decision tier is permitted to reach back into the profile store; the
contract makes that structural rather than a review note.
"""

from __future__ import annotations

from decider2 import Observed, module, observed, param, step
from decider2.types import Instant

from fraud_interdiction.tables.definitions import SEGMENTS, TEMPORAL


def client_segment_bits(client_id: str, event_timestamp: Instant) -> int:
    """46-bit mask of segment membership, resolved against the segment
    definition version in force at `event_timestamp`.

    Segments are not disjoint — a client is typically in 3–6 of 46. A bitmask
    means applicability is an `and` of two integers rather than a set operation,
    and the *definition* is versioned so the youth-age-limit change in spec
    §11.15 replays correctly against 1.9 M clients.
    """
    pass  # SEGMENTS.as_at(event_timestamp).membership(client_id)


def account_state_code(client_id: str, event_timestamp: Instant) -> int:
    """open / frozen / restricted / in-estate. Read live, not from the nightly
    snapshot — spec §4.2 marks this a hot field."""
    pass


def prior_confirmed_fraud_count(
    client_id: str,
    value: Observed[int] = observed(source="client_profile", stale_after="24h"),
) -> Observed[int]:
    """Count of prior confirmed-fraud events. Nightly snapshot.

    Returned as `Observed` — a client with no prior fraud and a client whose
    count could not be retrieved are different clients, and any rule reading
    this must be able to tell (spec §4.3, generalised to every source).
    """
    pass


def hours_since_contact_change(
    client_id: str,
    event_timestamp: Instant,
) -> Observed[float]:
    """Hours since the last contact-detail change. Feeds both the account-takeover
    family and the challenge-trust constraint in 5.13 — a client whose mobile
    number changed 40 minutes ago is not challenged by SMS OTP."""
    pass


def challenge_trust_bits(
    hours_since_contact_change: Observed[float],
    consent_bits: int,
    sms_distrust_hours: float = param(24.0, ge=0, le=168, unit="hours"),
) -> int:
    """Which challenge channels are *permitted* for this client right now.

    Distinguishes three states per channel, which the challenge matrix needs and
    a boolean cannot carry (spec §11.13): permitted, not-permitted-now (consent
    withdrawn or trust broken), not-configured-for-this-segment.
    """
    pass  # core.consent ∩ trust window


def legal_hold_active(client_id: str, event_timestamp: Instant) -> bool:
    """Court-ordered or investigation hold, as at the instant (spec §11.17).

    A temporal membership lookup, not a rule and not an overlay: a court order
    is an externally imposed interval over a named client, it reverts
    automatically at its end instant, and it must be answerable years later.
    Modelling it as an overlay would put a judge in the overlay register.
    """
    pass  # TEMPORAL.legal_holds.member_at(client_id, event_timestamp)


ClientContext = module(
    client_segment_bits,
    account_state_code,
    prior_confirmed_fraud_count,
    hours_since_contact_change,
    challenge_trust_bits,
    legal_hold_active,
    name="client_context",
    contract="contracts/feature_vector.json#/client_context",
)
