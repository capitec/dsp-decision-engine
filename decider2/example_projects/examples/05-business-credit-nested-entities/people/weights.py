"""Stage 5.8, PP-01 to PP-04 -- inclusion, coverage and weights.

The file that shows the **two-pass shape**, which is how a roll-up that needs a
total before it can compute an element's contribution is written without an
accumulator and therefore without an order dependency.

    Each(Entity, BaseWeight)        # per entity: a raw weight
    WeightTotals                    # Gather:     the totals
    Each(Entity, NormaliseWeight)   # per entity: reads the totals, broadcast down
    PeopleBlendFacts                # Gather:     the blend

PP-04 -- "where any single entity holds control or >= 50% effective ownership,
its weight is set to max(effective ownership, 0.60) and the remaining weights
are renormalised across the rest" -- is the reason the second pass exists. It is
also the reason a `Loop` with a carried accumulator would be wrong: the answer
depends on the whole collection, not on a prefix of it, so any sequential
formulation has to make two passes anyway and the second pass would be invisible
in the pipeline expression.
"""

from decider2 import module, param, step, Gather, count, sum_of, max_of, any_of, best_of, asc, desc
from grains import Entity, Application


# --------------------------------------------------------------------------
# PP-01 -- inclusion. Entity grain, one boolean, and three companions that keep
# the categories separate. PP-07 is the reason they are separate: "materiality
# exclusion is not an amnesty. An entity excluded from the blend by PP-01 is
# still subject to s5.4 and s5.6. A 2% shareholder with a confirmed fraud marker
# still declines the business. Exclusion is from the *average*, not from the
# *rules*."
#
# So `in_blend` gates the weight folds and nothing else. The disqualification
# folds in people/caps.py deliberately do not read it.
# --------------------------------------------------------------------------
def is_owner_entity(relationship_type_code: int, effective_ownership_pct: float) -> bool:
    """Holds equity, directly or effectively."""
    pass


def is_non_owning_controller(relationship_type_code: int, is_controlling: bool) -> bool:
    """Director, member, trustee or partner without equity."""
    pass


def in_blend(
    effective_ownership_pct: float,
    is_owner_entity: bool,
    is_non_owning_controller: bool,
    is_required_surety: bool,
    relationship_type_code: int,
    excluded_at_disqualification: bool,
    blend_inclusion_floor: float = param(
        5.0, ge=0.0, le=100.0,
        description="s6.1 blend inclusion floor. THIRD of the three parameters "
                    "that are 5.0 today -- the others are the expansion "
                    "materiality floor (structure/resolve.py) and nothing in "
                    "criticality.py. Change scenario 1 moves the expansion "
                    "floor to 3.0 and leaves this one alone."),
) -> bool:
    """PP-01. Sureties and guarantors are excluded here and handled by PP-08."""
    pass


def is_scoreable_owner(
    is_owner_entity: bool,
    in_blend: bool,
    counts_toward_coverage: bool,
) -> bool:
    """PP-02 numerator membership."""
    pass


Inclusion = module(
    is_owner_entity, is_non_owning_controller, in_blend, is_scoreable_owner,
    name="blend_inclusion", grain=Entity,
    taps=["in_blend", "is_scoreable_owner"],
)


# --------------------------------------------------------------------------
# PP-02 -- coverage. A Gather with a ratio taken at the Application grain.
#
# Note `coverage_weight_factor` -- thin-file entities count at HALF their
# ownership toward the numerator (s5.8 PP-02). That is a per-entity weight
# computed in entities/scoring/families.py and summed here, not a special case
# in the ratio.
# --------------------------------------------------------------------------
CoverageFacts = Gather(
    Entity, into=Application, name="people_coverage_facts",
    scoreable_ownership   = sum_of("covered_ownership_pct", where="is_scoreable_owner"),
    total_owner_ownership = sum_of("effective_ownership_pct", where="is_owner_entity"),
    owner_count           = count(where="is_owner_entity"),
    scoreable_owner_count = count(where="is_scoreable_owner"),
    unscoreable_material  = count(where="entity_unscoreable_and_material"),
    scoreable_person_count= count(where="is_scoreable_natural_person"),
)


def covered_ownership_pct(
    effective_ownership_pct: float,
    coverage_weight_factor: float,
) -> float:
    """Ownership discounted by the scoring situation. Entity grain."""
    pass


def people_coverage_ratio(
    scoreable_ownership: float,
    total_owner_ownership: float,
) -> float:
    """PP-02's ratio. Application grain."""
    pass


def insufficient_people_coverage(
    people_coverage_ratio: float,
    owner_count: int,
    scoreable_owner_count: int,
    coverage_requirement: float = param(
        75.0, ge=0.0, le=100.0,
        description="s6.1 coverage requirement. Policy, quarterly. "
                    "Acceptance criterion 13 names this as one of the three "
                    "values a Policy analyst must be able to change without an "
                    "engineer."),
) -> bool:
    """Below the requirement, or a single-owner company whose owner is
    unscoreable. Reason 5401, mandatory committee."""
    pass


def minimum_composition_fails(
    scoreable_person_count: int,
    segment_code: int,
) -> bool:
    """PP-10: at least one scoreable natural person for micro and small.
    Zero is a referral, reason 5403, regardless of the commercial view."""
    pass


PeopleCoverage = (
    module(covered_ownership_pct, name="covered_ownership", grain=Entity)
    | CoverageFacts
    | module(people_coverage_ratio, insufficient_people_coverage,
             minimum_composition_fails,
             name="people_coverage", grain=Application,
             taps=["people_coverage_ratio", "insufficient_people_coverage"])
)


# --------------------------------------------------------------------------
# PP-03 / PP-04 -- the two passes.
# --------------------------------------------------------------------------
def raw_weight(
    in_blend: bool,
    is_owner_entity: bool,
    is_non_owning_controller: bool,
    effective_ownership_pct: float,
    controller_notional_points: float = param(
        10.0, ge=0.0, le=100.0,
        description="s6.1: non-owning controllers receive a notional 10 points"),
) -> float:
    """PP-03 pass one: ownership for owners, a notional allocation for
    controllers. Not yet capped and not yet normalised."""
    pass


BaseWeight = module(raw_weight, name="base_weight", grain=Entity)


WeightTotals = Gather(
    Entity, into=Application, name="weight_totals",
    raw_weight_total        = sum_of("raw_weight", where="in_blend"),
    controller_points_total = sum_of("raw_weight", where="is_non_owning_controller"),
    max_ownership           = max_of("effective_ownership_pct", where="in_blend"),
    has_controller          = any_of("is_controlling", where="in_blend"),

    # PP-04 needs to know WHICH entity dominates, not only that one does, because
    # PP-06's "any included entity with grade >= 9 holding control caps the people
    # grade at that entity's grade" needs that entity's grade at the Application
    # grain. `best_of` lifts it.
    dominant_entity = best_of(
        "effective_ownership_pct",
        where="is_controlling_or_majority",
        tie_break=(desc("is_controlling"), asc("entity_key")),
        lift=["entity_id", "entity_key", "risk_grade", "effective_ownership_pct"],
    ),
)


def controller_points_applied(
    controller_points_total: float,
    controller_points_cap: float = param(
        30.0, ge=0.0, le=100.0,
        description="s6.1: collectively capped at 30 points"),
) -> float:
    """PP-03's collective cap on non-owning controllers. Application grain."""
    pass


def normalised_weight(
    raw_weight: float,
    raw_weight_total: float,
    controller_points_total: float,
    controller_points_applied: float,
    is_non_owning_controller: bool,
) -> float:
    """PP-03 pass two: scale down the controller allocation to its collective cap,
    then normalise so weights sum to 1. Entity grain, reading two broadcast
    totals."""
    pass


def blend_weight(
    normalised_weight: float,
    is_dominant_entity: bool,
    has_controller: bool,
    max_ownership: float,
    effective_ownership_pct: float,
    control_weight_floor: float = param(
        0.60, ge=0.0, le=1.0,
        description="s6.1 controlling entity weight floor. 'A business run by "
                    "one person is that person's risk, whatever the shareholder "
                    "register says.'"),
    control_ownership_threshold: float = param(50.0, ge=0.0, le=100.0),
) -> float:
    """PP-04: the dominant entity takes max(ownership, 0.60) and the rest
    renormalise across the remainder."""
    pass


def blend_weight_rule_applied(
    is_dominant_entity: bool,
    has_controller: bool,
    is_non_owning_controller: bool,
    controller_points_total: float,
    controller_points_applied: float,
) -> int:
    """Which of PP-03 / PP-04 set this entity's weight.

    s5.8's "Records" clause: "the full weight vector. 'Why did this business
    grade move from 6 to 7 when nothing changed?' is answered by the weight
    vector and the overlay decomposition together, and by nothing else."
    So the weight *and its cause* are per-entity outputs.
    """
    pass


NormaliseWeight = (
    module(controller_points_applied, name="controller_cap", grain=Application)
    | module(normalised_weight, blend_weight, blend_weight_rule_applied,
             name="blend_weights", grain=Entity,
             taps=["blend_weight", "blend_weight_rule_applied"])
)
