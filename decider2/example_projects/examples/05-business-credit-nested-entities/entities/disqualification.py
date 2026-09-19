"""Stage 5.4b -- the fourteen entity disqualification rules, three dispositions
each, and the injection of "material adverse event" dispositions back down into
the Event grain.

Two things here that doc 03 has no shape for.

1. **The disposition is a table lookup, not a constant.** A rule's outcome
   depends on the criticality class of the entity it fires on: 14 rules x 3
   classes = 42 cells, and s5.4 calls that matrix "a policy artefact". So
   `fires(..., gives=by_class(...))` reads the cell. Encoding it as 42 `if`s
   would make change scenario 4 ("for peripheral entities only") a code change,
   and would hide the asymmetry the spec insists must be readable.

2. **Three rules decline regardless of class, and s5.4 says that asymmetry
   "must be readable as deliberate, not as an oversight in a matrix".** So the
   rule declares `class_invariant=True` and the framework *asserts* it against
   the table at build time. A later edit that softens E-AROD-12 for peripheral
   entities fails the build with the rule id, rather than quietly changing the
   Bank's policy on write-offs.
"""

from decider2 import module, step, param, verdict, fires, table, by_class, emit_into
from grains import Entity, Event, PERIPHERAL, SIGNIFICANT, CRITICAL
from grains import CLEAR, REFER, DECLINE, DISPOSITION_ORDER

# Dispositions this matrix may contain. "Material adverse event" is not an
# outcome -- it is a *redirection* into the Event grain (see SyntheticEvents).
EXCLUDE_ENTITY, MATERIAL_ADVERSE_EVENT, RECORD_ONLY = 3, 4, 5

DISPOSITION_MATRIX = table(
    "entity_disqualification_disposition",
    key=("rule_id", "criticality_class"),
    columns=("disposition_code", "reason_code", "requires_committee"),
    source="tables/entity_disposition.csv",
    effective_dated=True,
    owner="Business Credit Risk Policy",
    cadence="quarterly",
)


def e_arod_01_fires(identity_verification_code: int) -> bool:
    """Identity verification failed."""
    pass


def e_arod_02_fires(
    screening_outcome_code: int,
    screening_confidence: float = 0.0,
    confirmed_confidence_floor: float = param(95.0, ge=0.0, le=100.0),
) -> bool:
    """Screening confirmed at or above the confirmed confidence floor."""
    pass


def e_arod_03_fires(
    screening_outcome_code: int,
    screening_confidence: float = 0.0,
    probable_floor: float = param(80.0, ge=0.0, le=100.0),
    probable_ceiling: float = param(95.0, ge=0.0, le=100.0),
) -> bool:
    """Screening probable (80-94)."""
    pass


def e_arod_08_fires(under_debt_review: bool, debt_review_cleared: bool) -> bool:
    """Under debt review, not cleared."""
    pass


def e_arod_10_fires(
    age_at_final_instalment_years: float,
    is_required_surety: bool,
    surety_age_ceiling: float = param(75.0, ge=50.0, le=100.0),
) -> bool:
    """Age at final instalment above the ceiling, and acting as a surety."""
    pass


def e_arod_12_fires(
    months_since_bank_write_off: float,
    write_off_lookback_months: float = param(120.0, ge=0.0, le=240.0),
) -> bool:
    """Named on a Bank write-off, any capacity, within the lookback."""
    pass


def e_arod_14_fires(confirmed_fraud_marker: bool) -> bool:
    """Confirmed fraud marker."""
    pass


# ... seven more.


EntityDisqualification = verdict(
    name="entity_disqualification",
    of=Entity,
    writes="entity_disqualification_disposition",
    resolve=DISPOSITION_ORDER,
    collect="all",
    rules=[
        fires(e_arod_01_fires, gives=by_class(DISPOSITION_MATRIX, "E-AROD-01")),
        fires(e_arod_02_fires, gives=by_class(DISPOSITION_MATRIX, "E-AROD-02"),
              class_invariant=True),      # asserted against the table at build
        fires(e_arod_03_fires, gives=by_class(DISPOSITION_MATRIX, "E-AROD-03")),
        fires(e_arod_08_fires, gives=by_class(DISPOSITION_MATRIX, "E-AROD-08")),
        fires(e_arod_10_fires, gives=by_class(DISPOSITION_MATRIX, "E-AROD-10")),
        fires(e_arod_12_fires, gives=by_class(DISPOSITION_MATRIX, "E-AROD-12"),
              class_invariant=True),
        fires(e_arod_14_fires, gives=by_class(DISPOSITION_MATRIX, "E-AROD-14"),
              class_invariant=True),
        # ... seven more
    ],
    interior="config/business_facility/rules/entity_disqualification.json",
    contract="contracts/entity_disqualification.json",
)


# --------------------------------------------------------------------------
# s5.4: "Where the disposition is 'material adverse event', the finding does not
# decline here; it is injected into that entity's event list at s5.5 with a
# synthetic event so that it participates in the count-based and amount-based
# roll-up rules rather than sitting outside them. Synthetic events are marked as
# such."
#
# That is a grain shift *downward from a rule outcome* -- Entity to Event -- and
# it is the one that doc 03 has no story for at all. `emit_into` is the inverse
# of `Gather`: a parent-grain rule appends rows to the child frame, under a
# declared schema, with provenance.
#
# Two properties make it safe rather than a back door:
#   * the emitted rows are marked `is_synthetic` and carry `source_rule_id`, so
#     every downstream count can be reported with and without them;
#   * the emission happens strictly *before* Each(Event, ClassifyEvent), which
#     is visible in the pipeline expression, so there is no question of a fold
#     having already run. The framework rejects an `emit_into` that would
#     invalidate a Gather already computed at that grain.
# --------------------------------------------------------------------------
SyntheticEvents = emit_into(
    Event,
    from_grain=Entity,
    when="entity_disqualification_disposition == MATERIAL_ADVERSE_EVENT",
    fields={
        "event_type_code": "synthetic_event_type_for_rule",   # a registered step
        "event_date": "decision_date",
        "amount": "null",
        "status_code": "ACTIVE",
        "is_synthetic": True,
        "source_rule_id": "entity_disqualification_binding_rule",
    },
    name="synthetic_events",
)
