"""The field catalogue — which of the ~210 event fields exist on which event
type, from when, and in what unit.

Spec §4.1: "A rule referencing a field its event type does not carry is a
definition error that must be caught before deployment, not at 02:00 on a
Sunday."

This is the only place that sentence is enforceable, so it is **code**, not a
rule document. A rule document may *reference* a field; it may never *declare*
one. That asymmetry is the whole point: analysts add rules freely and cannot
invent inputs.

Deviation from doc 03: doc 03 infers a module's interface from parameter names
and validates it against one input-frame schema (doc 07 §5, `build --schema`).
One schema cannot express twelve event types whose field sets differ by 30
fields, nor a field that did not exist before 2026-04-01 (spec §11.6). The
catalogue is a *variant-aware, effective-dated* schema. See
FRAMEWORK-DEMANDS.md #4 and #5.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from decider2.schema import FieldCatalogue, FieldRow, Variant, VariantSet

# --- the twelve assessable event types (spec §4.1) -------------------------

EVENT_TYPES = VariantSet(
    name="event_type_code",
    dtype="int16",
    variants=[
        Variant(110, "card_auth_present",   latency_ceiling_ms=25),
        Variant(111, "card_auth_not_present", latency_ceiling_ms=25),
        Variant(210, "instant_payment",     latency_ceiling_ms=25),
        Variant(211, "eft_credit",          latency_ceiling_ms=400),
        Variant(220, "debit_order_dispute", latency_ceiling_ms=400),
        Variant(310, "login",               latency_ceiling_ms=40),
        Variant(311, "device_registration", latency_ceiling_ms=40),
        Variant(312, "sim_change",          latency_ceiling_ms=40, may_arrive_late=True),
        Variant(313, "beneficiary_add",     latency_ceiling_ms=40),
        Variant(314, "contact_change",      latency_ceiling_ms=40),
        Variant(315, "limit_change",        latency_ceiling_ms=40),
        Variant(316, "card_reissue",        latency_ceiling_ms=400),
    ],
)


def build_catalogue(rows: Iterable[FieldRow]) -> FieldCatalogue:
    """Validate catalogue rows and return the frozen catalogue.

    Raises on: a field declared twice; a field carried by no variant; a
    `derived_from` naming an unregistered feature; an `available_from` later
    than an `available_to`.

    The rows arrive from the caller's loader (doc 08 §6 — the framework does not
    fetch documents). `events/field_catalogue.csv` is the checked-in default.
    """
    pass  # validate rows, freeze, index by (field_name, variant, instant)


CATALOGUE: FieldCatalogue = build_catalogue.deferred(  # bound at startup by the loader
    variants=EVENT_TYPES,
    name="fraud_event_fields",
    contract="contracts/feature_vector.json",
)


def assert_rule_fields_available(
    catalogue: FieldCatalogue,
    rule_id: str,
    event_types: frozenset[int],
    referenced: frozenset[str],
    effective_from: datetime,
) -> None:
    """Static check run at ruleset validation, before any compile.

    The error this exists to produce:

        MS-0208 references 'three_d_secure_outcome'. That field is carried by
        event types 111 only; the rule declares {210, 211, 313}. Either narrow
        the rule's event types to {111} or reference a field common to all four.

        MS-0311 references 'merchant_initiated_indicator', available from
        2026-04-01. The rule is effective from 2026-02-14, so on 46 days of its
        own effective window the field does not exist. Set effective_from to
        2026-04-01, or declare on_absent: evaluate_false.
    """
    pass  # intersect declared event types with per-field availability; raise with the above
