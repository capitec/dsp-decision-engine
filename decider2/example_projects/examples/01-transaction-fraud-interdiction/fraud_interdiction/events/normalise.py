"""5.1 — event admission and normalisation.

Turns a raw inbound message into the 41 common fields plus the type-specific
fields, with every failed validation kept rather than dropped: "a malformed
message is itself a weak fraud signal and a dropped event is the one the dispute
is about" (spec §5.1).
"""

from __future__ import annotations

from decider2 import Observed, missing_as, module, param, step
from decider2.types import Instant, Money

from fraud_interdiction.events.catalogue import CATALOGUE


def event_age_seconds(event_timestamp: Instant, assessment_timestamp: Instant) -> float:
    """Seconds between when the event happened and when this engine answered."""
    pass  # (assessment - event).total_seconds()


def event_age_flag(
    event_age_seconds: float,
    event_type_code: int,
    late_tolerance_s: float = param(300.0, ge=0, le=86400, unit="seconds"),
) -> int:
    """0 fresh, 1 late-but-expected (SIM change), 2 late-and-anomalous.

    A card authorisation is never legitimately five minutes old; a SIM change
    routinely is. The distinction is a variant property, not a threshold.
    """
    pass  # consult EVENT_TYPES[event_type_code].may_arrive_late


def amount_zar_cents(
    amount_minor: Money = missing_as(0),
    fx_rate_used: float = missing_as(1.0),
) -> Money:
    """Amount in the reporting currency, converted at the rate in the message.

    int64 cents throughout. The rate used is recorded, never re-fetched: a
    replay that reaches for today's rate has failed (spec §9.2).
    """
    pass  # round_half_up(amount_minor * fx_rate_used) — never bare round()


def mandatory_fields_present(event_type_code: int) -> bool:
    """Every field the catalogue marks non-nullable for this variant is present."""
    pass  # CATALOGUE.mandatory(event_type_code) ⊆ populated


def admission_defect_code(
    mandatory_fields_present: bool,
    event_age_flag: int,
    field_validation_failures: int,
) -> int:
    """Bitset of admission defects. Non-zero narrows the applicable population
    rather than dropping the event."""
    pass


Normalise = module(
    event_age_seconds,
    event_age_flag,
    amount_zar_cents,
    mandatory_fields_present,
    admission_defect_code,
    name="normalise",
    catalogue=CATALOGUE,
    taps=["admission_defect_code", "event_age_flag"],
)
