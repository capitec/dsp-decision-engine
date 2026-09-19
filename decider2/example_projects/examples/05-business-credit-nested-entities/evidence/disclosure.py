"""Two reason sets, and the boundary between them.

s9.2: "The business is entitled to the reason for ITS decline. It is generally
not entitled to the details of a third party's credit record -- a director's
personal judgment is that director's information. Getting this backwards is a
privacy breach in one direction and a regulatory failure to give reasons in the
other."

s13 Q14: "How do two reason sets coexist without the communicable one being
derived by hand, and without the internal one leaking?"

--------------------------------------------------------------------------
THE MECHANISM
--------------------------------------------------------------------------
A **disclosure class** on every output, defaulting to INTERNAL, and a registry
attribute on every reason code. The client payload is assembled by projection
over the disclosure class -- so a new value is invisible to the client until
somebody classifies it, rather than visible until somebody hides it.

doc 04 s5.3 already establishes that per-record diagnostics are PII by
construction. This project needs that made operational rather than documented,
because the third-party boundary is *inside* one decision record: the applicant
is entitled to some of it and not to the rest, and the line runs along the
grain boundary. Everything at the Entity and Event grain is a third party's
information. FRAMEWORK-DEMANDS D20.
"""

from decider2 import module, step, table, disclosure, projection
from grains import Application, Entity, Event

INTERNAL, BUSINESS, INDIVIDUAL = 1, 2, 3

REASON_REGISTRY = table(
    "reason_codes",
    key="reason_code",
    columns=("severity_rank", "communicable_at_business_level",
             "client_wording_en", "client_wording_af", "client_wording_zu",
             "regulatory_classification", "effective_from", "effective_to"),
    source="core.reason_codes",           # the library's registry, not a copy
    effective_dated=True,
    owner="Compliance",
    cadence="monthly",
)


# --------------------------------------------------------------------------
# The default. A grain-level declaration, so that adding a value at the Entity
# grain cannot make it client-visible by accident.
# --------------------------------------------------------------------------
DISCLOSURE_DEFAULTS = disclosure.defaults({
    Application: BUSINESS,
    Entity:      INTERNAL,
    Event:       INTERNAL,
})

# The named exceptions -- Application-grain values that are NOT for the client,
# and the tiny set of Entity-grain values that are. Both lists are short and
# both are reviewed by Compliance rather than by engineering.
DISCLOSURE_OVERRIDES = disclosure.overrides({
    "people_pd":                        INTERNAL,
    "people_pd_unadjusted":             INTERNAL,
    "people_pd_overlay_contribution":   INTERNAL,
    "adjustments_applied":              INTERNAL,
    "attributing_event_ids":            INTERNAL,
    "attribution_chain":                INTERNAL,
    # An entity may obtain their OWN reasons directly. The route is disclosable
    # to the business; the content is not.
    "individual_reason_route_code":     BUSINESS,
})


def communicable_reason_codes(
    fired_reason_mask: int,
    decision_date: int,
) -> list[int]:
    """The client-facing set: every fired reason whose registry entry is
    communicable at business level, ranked by severity.

    Where a decline was caused by a third party's record, the registry supplies a
    generic code -- "an individual associated with the business has adverse
    credit information" -- and the route by which that individual obtains their
    own reasons. The substitution is a registry attribute, not a step in this
    project, which is what stops 140 new reason codes drifting from their
    disclosure classification.
    """
    pass


def internal_reason_codes(fired_reason_mask: int, decision_date: int) -> list[int]:
    """The complete set, ranked, with the entity and event attribution attached."""
    pass


def primary_reason_code(communicable_reason_codes: list[int]) -> int:
    """The one communicated to the client. Ranked through core.reason_codes.

    Determinism (s8) requires the ORDER of decline_reason_codes to be
    reproducible, which means the severity rank must be total. Two codes at the
    same rank are broken by code number -- declared in the registry, not here.
    """
    pass


def individual_reason_route_code(attributing_entity_id: int | None) -> int:
    """How the individual whose record caused the decline obtains their own
    reasons. Disclosable to the business; the reasons themselves are not."""
    pass


ReasonSets = module(
    communicable_reason_codes, internal_reason_codes,
    primary_reason_code, individual_reason_route_code,
    name="reason_sets", grain=Application,
)


# --------------------------------------------------------------------------
# The two payloads. Projections over the record, not two assemblies of it.
# --------------------------------------------------------------------------
CLIENT_PAYLOAD = projection(
    "client_decision_payload",
    disclosure_at_most=BUSINESS,
    include_grains=[Application],
)

INTERNAL_RECORD = projection(
    "internal_decision_record",
    disclosure_at_most=INTERNAL,
    include_grains=[Application, Entity, Event, "candidate", "period"],
    retention_years=7,
)

# s10 acceptance criterion 15 -- "the client-facing reason set never discloses a
# third party's event detail, and the internal attribution always contains it" --
# is then a property of the projection and is testable without running the flow:
# assert no Entity-grain or Event-grain value reaches CLIENT_PAYLOAD.
