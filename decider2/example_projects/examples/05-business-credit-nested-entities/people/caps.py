"""Stage 5.8, PP-06 and PP-07 -- the worst-of overrides, applied AFTER the blend
and dominating it.

Five rules, four of which need to name the entity that bound them. That is what
makes this file the place where **attribution crosses the second grain
boundary**: a business-level cap has to carry an Entity identity, and the entity
it names already carries an Event set from s5.6.

The chain is:

    PP-06 fires  ->  witness = {entity 7}
    entity 7     ->  binding_rule_id = AE-R-01, witness = {event 41}
    event 41     ->  classifying_rule_id = AE-C-03, thresholds, overlay id

Nobody assembles that. Each grain records its own (rule, quantity, witness)
triple, and `evidence/attribution.py` joins them by key. Spec s13 Q4 -- "does
the framework carry attribution upward, or does each project rebuild it?" -- is
answered by the witness, and the answer is that no level knows about the level
below it.
"""

from decider2 import module, param, step, Gather, count, any_of, best_of, witness
from decider2 import verdict, fires, asc, desc
from grains import Entity, Application, DISQUALIFYING


# --------------------------------------------------------------------------
# A Gather over the SAME Entity collection as people/weights.py and
# people/blend.py, with different predicates. Third roll-up of the same
# elements.
#
# PP-07 is enforced here by what these folds do NOT filter on: none of them
# reads `in_blend`. A 2% shareholder is excluded from the average and included
# in the rules, which is exactly s5.8's "exclusion is from the *average*, not
# from the *rules*."
# --------------------------------------------------------------------------
PeopleCapFacts = Gather(
    Entity, into=Application, name="people_cap_facts",

    # PP-06 row 1 -- any INCLUDED entity disqualifying. This one does read
    # in_blend, because PP-06 says "included"; the s5.6 disqualification of an
    # excluded entity reaches the outcome through validation/consistency.py's
    # FV-08 instead, and the two routes are deliberately different.
    disqualifying_entity_count = count(
        where="in_blend_and_adverse_disqualifying"),
    binding_disqualifying_entity = best_of(
        "entity_adverse_verdict_code",
        where="in_blend_and_adverse_disqualifying",
        tie_break=(desc("effective_ownership_pct"), asc("entity_key")),
        lift=["entity_id", "entity_key", "binding_rule_id",
              "effective_ownership_pct", "criticality_class"],
    ),

    # PP-06 row 2 -- grade 11/12 holding >= 20%.
    severe_grade_major_count = count(where="is_severe_grade_major_owner"),
    binding_severe_owner = best_of(
        "risk_grade", where="is_severe_grade_major_owner",
        tie_break=(desc("effective_ownership_pct"), asc("entity_key")),
        lift=["entity_id", "entity_key", "risk_grade", "effective_ownership_pct"],
    ),

    # PP-06 row 3 -- grade >= 9 holding control. Caps at THAT entity's grade,
    # so the grade has to be lifted, not just counted.
    binding_controller = best_of(
        "risk_grade", where="is_weak_controller",
        tie_break=(desc("risk_grade"), asc("entity_key")),
        lift=["entity_id", "entity_key", "risk_grade"],
    ),

    # PP-06 row 4 -- two or more entities at grade >= 9 together holding >= 40%.
    # A count AND a sum, and the witness of either names the whole group. This
    # is the clearest case of s5.6's "count rules attribute to a SET".
    weak_grade_count      = count(where="is_weak_grade_entity"),
    weak_grade_ownership  = sum_of("effective_ownership_pct",
                                   where="is_weak_grade_entity"),

    # PP-06 row 5 -- recent_adverse on any critical entity.
    critical_recent_adverse = any_of("recent_adverse_and_critical"),

    # PP-02's referral inputs and the excluded-entity list required by s5.8's
    # "Emits": the excluded set with reasons.
    excluded_entity_count = count(where="excluded_from_blend"),
)


def pp_06_disqualifying_fires(disqualifying_entity_count: int) -> bool:
    """Any included entity with a disqualifying adverse verdict. Reason 5402."""
    pass


def pp_06_severe_owner_fires(
    severe_grade_major_count: int,
    severe_ownership_floor: float = param(20.0, ge=0.0, le=100.0),
) -> bool:
    """Grade 11 or 12 with at least 20% effective ownership. Caps at grade 10."""
    pass


def pp_06_weak_controller_fires(binding_controller_risk_grade: int,
                                weak_grade_floor: int = param(9, ge=1, le=12)) -> bool:
    """A controlling entity at grade 9 or worse. Caps at that entity's grade."""
    pass


def pp_06_weak_group_fires(
    weak_grade_count: int,
    weak_grade_ownership: float,
    weak_group_min_count: int = param(2, ge=2, le=10),
    weak_group_min_ownership: float = param(40.0, ge=0.0, le=100.0),
) -> bool:
    """Two or more weak-grade entities together holding 40% or more. Worsens by 2."""
    pass


def pp_06_recent_adverse_fires(critical_recent_adverse: bool) -> bool:
    """recent_adverse on any critical entity. People grade may not beat 5."""
    pass


PeopleCaps = (
    PeopleCapFacts
    | verdict(
        name="people_caps",
        of=Application,
        writes="people_grade_capped",
        resolve="worst",                  # a cap ordering, not a severity one
        collect="all",
        rules=[
            fires(pp_06_disqualifying_fires, gives="business_decline", reason=5402,
                  attributes=witness("disqualifying_entity_count"),
                  names_entity="binding_disqualifying_entity"),

            fires(pp_06_severe_owner_fires, caps_grade_at=10,
                  attributes=witness("severe_grade_major_count"),
                  names_entity="binding_severe_owner"),

            fires(pp_06_weak_controller_fires,
                  caps_grade_at="binding_controller_risk_grade",
                  names_entity="binding_controller"),

            fires(pp_06_weak_group_fires, worsens_grade_by=2,
                  attributes=witness("weak_grade_count"),
                  quantity="weak_grade_ownership"),

            fires(pp_06_recent_adverse_fires, caps_grade_at=5,
                  attributes=witness("critical_recent_adverse")),
        ],
        interior="config/business_facility/rules/people_caps.json",
    )
)

# --------------------------------------------------------------------------
# `names_entity=` is the one attribution clause that is specific to a grain
# shift: it says "this cap's attribution is an Entity identity, and here is the
# lifted record". Without it a business decline carries a witness bitmask over
# entities and the record assembler has to guess which one to name as
# `attributing_entity_id` -- the output s7.1 calls "the output the project
# exists for". With it, s10 acceptance criterion 1 (zero exceptions across 5 000
# applications) is a build-time property: a `fires(gives="business_decline")`
# without `names_entity` or `attributes` fails the build.
# --------------------------------------------------------------------------
