"""Counterfactuals -- three of them, one construct.

`Shadow` re-evaluates a declared sub-pipeline under a declared perturbation and
suffixes the outputs. It appears four times in this project:

    grade/overlay_stack.py   UnadjustedSpine      overlays off       always on
    here                     DisputeShadow        provisional off    always on
    here                     EventCounterfactual  one event masked   on demand
    here                     SmallestChange       amount/term/event  on demand

Four uses is what makes it a construct rather than a special case, and each one
is a requirement the spec states separately without noticing they are the same
shape. That is the argument for it being framework rather than project code.
"""

from decider2 import Shadow, mask, overlays_off, module, param, step, param_key
from decider2 import Gather, best_of, count, asc, desc
from grains import Application, Entity, Event, Candidate

from entities.adverse.classify import ClassifyEvent
from entities.adverse.gather import EntityAdverseFacts
from entities.adverse.verdict import EntityAdverseVerdict
from people.blend import PeopleBlend
from people.caps import PeopleCaps
from grade.combine import BusinessGrade
from pricing.select import SelectOffer
from validation.consistency import FinalValidation

NESTED_SPINE = (ClassifyEvent | EntityAdverseFacts | EntityAdverseVerdict
                | PeopleBlend | PeopleCaps | BusinessGrade)


# --------------------------------------------------------------------------
# 1. The dispute shadow. ALWAYS ON, because AE-C-22's consequence is a rule of
#    the flow and not a report:
#
#    "A disputed event may never be the sole cause of a decline. Where removing
#     every provisional classification would change the outcome from decline to
#     anything else, the outcome becomes a referral with the dispute identified.
#     The Bank does not decline a business on a record the individual is
#     formally contesting."
#
#    The only way to know whether removing the provisional classifications would
#    change the outcome is to remove them and look. Computing it always is
#    expensive and computing it on demand is impossible, because the demand is
#    the outcome.
#
#    `share_prefix=True` is doing real work here: the perturbation is at the
#    Event grain and reaches only entities that HAVE a disputed event, which is
#    a small minority. Static lineage proves the rest of the Entity frame is
#    unaffected, so the shadow is proportional to the disputed population rather
#    than to the batch. Without that proof this shadow doubles the cost of the
#    most expensive stage in the flow for every application, and the honest
#    consequence would be that the Bank cannot afford its own policy.
#    FRAMEWORK-DEMANDS D18.
# --------------------------------------------------------------------------
DisputeShadow = Shadow(
    NESTED_SPINE | SelectOffer | FinalValidation,
    perturb=mask(Event, where="classification_provisional"),
    suffix="_without_disputes",
    share_prefix=True,
    name="dispute_shadow",
    emits=["outcome_code_without_disputes",
           "entity_adverse_verdict_code_without_disputes"],
)


def dispute_would_change_outcome(
    outcome_code: int, outcome_code_without_disputes: int
) -> bool:
    """A decline that becomes anything else without the disputed events."""
    pass


def disputed_event_ids(entity_provisional_witness: int) -> list[int]:
    """"...with the dispute identified." The witness of the provisional fold,
    resolved to event ids, so the referral names the events rather than saying
    that some exist."""
    pass


DisputeOutcome = module(
    dispute_would_change_outcome, disputed_event_ids,
    name="dispute_outcome", grain=Application,
    taps=["dispute_would_change_outcome"],
)


# --------------------------------------------------------------------------
# 2. The single-event counterfactual. s9.4, ON DEMAND, and it must be available
#    "for any event, not only the binding one, because a dispute usually
#    concerns the event the individual cares about rather than the event the
#    rule fired on".
#
#    Four answers required: how it was classified; what the ENTITY verdict would
#    have been without it; what the PEOPLE and BUSINESS grades would have been;
#    what the OFFER would have been. Those are the four suffixed outputs of one
#    shadow over four stages.
#
#    "on the artefacts in force at the original decision_date" is free: every
#    table in this project is effective-dated on `decision_date`, which is an
#    input, and never on "today". Replaying with the same inputs resolves the
#    same versions. s7.3 of the core library, made operational by there being no
#    other way to read a table.
# --------------------------------------------------------------------------
EventCounterfactual = Shadow(
    NESTED_SPINE | SelectOffer,
    perturb=mask(Event, key=param_key("counterfactual_event_id")),
    suffix="_counterfactual",
    on_demand=True,
    share_prefix=True,
    name="event_counterfactual",
    emits=["entity_adverse_verdict_code_counterfactual",
           "people_grade_counterfactual",
           "risk_grade_counterfactual",
           "offered_amount_counterfactual",
           "binding_constraint_code_counterfactual"],
)


# --------------------------------------------------------------------------
# 3. Change scenario 15 -- "credit committee wants the pack to show, for every
#    declined application, the smallest change that would have made it
#    approvable: which amount, which term, which entity's event removed, or
#    which additional security."
#
#    The spec's own §13 Q17 says this "is a good test of whether the pricing
#    stage was expressed as data or as control flow."
#
#    Expressed as data, most of it is already computed:
#      * which amount / which term  -> `next_larger_blocked` in pricing/select.py,
#        which is a fold over candidates that were evaluated anyway
#      * which additional security  -> a Shadow over the Candidate grain with the
#        cover uplifted, which re-prices the SAME enumerated space
#      * which entity's event removed -> the event counterfactual above, folded
#        over the events of the binding entity
#
#    So the "smallest change" is a fold over three shadows rather than a new
#    search. That is the test passing.
# --------------------------------------------------------------------------
SecurityUpliftShadow = Shadow(
    SelectOffer,
    perturb=mask.uplift("adjusted_collateral_value",
                        by=param_key("uplift_amount")),
    suffix="_with_security",
    on_demand=True,
    name="security_uplift_shadow",
)

EventRemovalSweep = Gather(
    Event, into=Application, name="event_removal_sweep",
    # One counterfactual per event on the binding entity, folded to the one
    # whose removal most improves the outcome. Bounded by Event.capacity, so
    # the sweep is at most 60 evaluations and is declared to be.
    best_single_removal = best_of(
        "outcome_improvement_counterfactual",
        where="is_on_binding_entity",
        tie_break=(desc("event_severity_code"), asc("event_id")),
        lift=["event_id", "event_type_code", "amount",
              "outcome_code_counterfactual", "offered_amount_counterfactual"],
    ),
    on_demand=True,
)


def smallest_approvable_change_code(
    next_larger_blocked_candidate_amount: float,
    next_larger_blocked_binding_constraint_code: int,
    best_single_removal_event_id: int | None,
    security_uplift_required: float,
) -> int:
    """Which of the four levers is smallest. Ranked by a declared order --
    reduce amount, shorten term, remove an event, add security -- because
    "smallest" is not comparable across levers and pretending it is would give
    the committee a number with no meaning."""
    pass


SmallestChange = module(
    smallest_approvable_change_code,
    name="smallest_change", grain=Application, on_demand=True,
)
