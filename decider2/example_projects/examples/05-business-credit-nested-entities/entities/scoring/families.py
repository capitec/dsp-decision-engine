"""Stage 5.7 -- two scorecard families over one heterogeneous collection.

Spec s13 Q9: "How are two scorecard families with different characteristic sets
applied to a single heterogeneous collection, where the choice depends on an
attribute of the element, and both must contribute to one blended result?"

The answer needs nothing new. It is `Branch` (doc 03 s8.2) *inside* `Each` --
a branch at the Entity grain on `is_natural_person`, with the two arms reading
entirely different leaf inputs and producing the same `modifies` set.

Three things that are not obvious and that the spec forces:

1. **The arms have different inputs, not just different constants.** BUS-PERS-01
   reads 38 consumer-bureau characteristics; BUS-COMM-01 reads 29 commercial
   ones. doc 03 s8.2 requires every arm to produce every declared `modifies`
   value with agreeing types -- it says nothing about inputs, and nothing needs
   to. Only the taken arm executes, so a juristic entity never demands a
   `residential_stability_months` column. But the *frame* must still carry both
   sets of columns, mostly null, which is a real cost at the Entity grain and is
   D22 in FRAMEWORK-DEMANDS.

2. **Four scoring situations, not two.** s5.7 is explicit that scored / thin
   file / no hit / no enquiry possible must not collapse, and that the
   difference between the last two "is exactly the one that gets lost". So the
   situation is a 4-way router, nested inside the family branch. Eight arms.

3. **A scorecard-scoped overlay applied to the wrong family is an error, not a
   no-op.** s5.7: "applying a BUS-PERS-01-scoped overlay to a juristic entity is
   an error, not a silent no-op, and the flow must fail loudly rather than
   quietly scoring an entity with the wrong stack." That is a property of
   `overlay.position(...)`'s scope declaration and is checked per entity at
   resolution, not per application. FRAMEWORK-DEMANDS D17.
"""

from decider2 import module, param, step, Branch, scorecard, overlay, table
from grains import Entity

SCORED, THIN_FILE, NO_HIT, NO_ENQUIRY = 1, 2, 3, 4


# --------------------------------------------------------------------------
# The situation router. Entity grain, four outcomes, and the distinction
# between NO_HIT and NO_ENQUIRY is the point of the whole step.
# --------------------------------------------------------------------------
def scoring_situation_code(
    bureau_response_valid: bool,
    bureau_record_present: bool,
    consent_present: bool,
    identity_verification_code: int,
    bureau_errored: bool,
    active_account_count: int,
    months_of_history: float,
    scored_account_floor: int = param(3, ge=1, le=10),
    scored_history_months: float = param(12.0, ge=0.0, le=60.0),
) -> int:
    """s5.7's four situations. NO_HIT is a valid bureau response with no record;
    NO_ENQUIRY is consent absent, identity unverified, or a bureau error."""
    pass


def counts_toward_coverage(scoring_situation_code: int) -> bool:
    """PP-02's numerator membership. Thin file counts at half weight; no hit and
    no enquiry do not count at all."""
    pass


def coverage_weight_factor(
    scoring_situation_code: int,
    thin_file_factor: float = param(0.5, ge=0.0, le=1.0),
) -> float:
    """1.0 scored, 0.5 thin file, 0.0 otherwise."""
    pass


def entity_unscoreable(scoring_situation_code: int) -> bool:
    """s5.7: above 10% ownership this forces a referral."""
    pass


Situation = module(
    scoring_situation_code, counts_toward_coverage,
    coverage_weight_factor, entity_unscoreable,
    name="scoring_situation", grain=Entity,
    taps=["scoring_situation_code"],
)


# --------------------------------------------------------------------------
# The two families. `scorecard(...)` is the existing core component kind
# (doc 08 s3.4: bins -> points, uniform, generic kernel, free interior change).
# The overlays are declared on the scorecard, positions 2, 3 and 4 of the stack.
# --------------------------------------------------------------------------
PersonalScorecard = scorecard(
    "BUS-PERS-01",
    characteristics=38,
    source="tables/scorecards/bus_pers_01.json",
    calibration_segment="person-behind-business",
    effective_dated=True,
    overlays=[
        overlay.position(2, scopes=("scorecard_id", "segment_code", "is_new_to_bank")),
        overlay.position(3, scopes=("scorecard_id", "sector_grouping_code")),
        overlay.position(4, scopes=("is_natural_person",)),
    ],
    emits_contributions=True,          # s5.7: per-characteristic contributions are
                                       # required output, not a debugging nicety
)

PersonalScorecardThin = scorecard(
    "BUS-PERS-01T", characteristics=17,
    source="tables/scorecards/bus_pers_01t.json",
    calibration_segment="person-behind-business",
    grade_floor=7,                     # thin file cannot be better than grade 7
    effective_dated=True,
)

CommercialScorecard = scorecard(
    "BUS-COMM-01",
    characteristics=29,
    source="tables/scorecards/bus_comm_01.json",
    calibration_segment="commercial-sme",
    effective_dated=True,
    overlays=[
        overlay.position(2, scopes=("scorecard_id", "segment_code")),
        overlay.position(3, scopes=("scorecard_id", "sector_grouping_code")),
        overlay.position(4, scopes=("is_natural_person",)),
    ],
    emits_contributions=True,
)

CommercialScorecardThin = scorecard(
    "BUS-COMM-01T", characteristics=12,
    source="tables/scorecards/bus_comm_01t.json",
    calibration_segment="commercial-sme",
    grade_floor=7,
    effective_dated=True,
)

FALLBACK_GRADES = table(
    "no_hit_fallback_grades",
    key=("relationship_type_code", "age_band_index", "tenure_band_index"),
    columns=("risk_grade",),
    source="tables/fallback_grades.csv",
    effective_dated=True,
)


def fallback_grade(
    relationship_type_code: int,
    age_band_index: int,
    tenure_band_index: int,
) -> int:
    """s5.7: a no-hit entity gets a grade from role x age x tenure, and does not
    count toward coverage."""
    pass


def young_entity_grade_floor(
    months_since_registration: float,
    is_natural_person: bool,
    young_entity_months: float = param(12.0, ge=0.0, le=60.0),
    young_entity_floor: int = param(8, ge=1, le=12),
) -> int:
    """A juristic entity registered less than 12 months before decision_date is
    floored at grade 8 regardless of its commercial bureau result."""
    pass


def recent_adverse_grade_floor(
    recent_adverse: bool,
    recent_adverse_floor: int = param(4, ge=1, le=12),
) -> int:
    """AE-R-08's `recent_adverse` blocks entity grades 1-3."""
    pass


# --------------------------------------------------------------------------
# Composition. Two nested branches: family, then situation.
#
# The eight arms are four real scorecards, two thin variants, one fallback
# table and one no-op. `modifies` names the four values every arm must produce,
# and the four unadjusted counterparts, which is what makes PP-11's
# decomposition possible at all.
# --------------------------------------------------------------------------
def is_natural_person_entity(is_natural_person: bool) -> bool:
    """Route to the personal or commercial family."""
    pass


ScorePersonal = Branch(
    scoring_situation_code,
    [PersonalScorecard, PersonalScorecardThin, FallbackGrade, Unscoreable],
    modifies=["score", "score_unadjusted", "probability_of_default",
              "probability_of_default_unadjusted", "risk_grade",
              "risk_grade_unadjusted", "scorecard_id", "adjustments_applied"],
)

ScoreCommercial = Branch(
    scoring_situation_code,
    [CommercialScorecard, CommercialScorecardThin, FallbackGrade, Unscoreable],
    modifies=["score", "score_unadjusted", "probability_of_default",
              "probability_of_default_unadjusted", "risk_grade",
              "risk_grade_unadjusted", "scorecard_id", "adjustments_applied"],
)

ApplyGradeFloors = module(
    young_entity_grade_floor, recent_adverse_grade_floor,
    name="entity_grade_floors", grain=Entity,
)

EntityScoring = (
      Situation
    | Branch(is_natural_person_entity, ScorePersonal, ScoreCommercial,
             modifies=["score", "score_unadjusted", "probability_of_default",
                       "probability_of_default_unadjusted", "risk_grade",
                       "risk_grade_unadjusted", "scorecard_id",
                       "adjustments_applied"])
    | ApplyGradeFloors
)

# --------------------------------------------------------------------------
# Change scenario 6: "the commercial scorecard is replaced with a
# 34-characteristic version, and both must run in parallel for three months --
# the old one for existing clients and the new one for new-to-bank, in the same
# flow, on the same application where a group company is an existing client and
# the applicant is not."
#
# That is a third arm on the commercial router, keyed on an Entity-grain
# attribute. It is a skeleton change (one line here, one arm), not a
# per-application switch, and *because the routing key is an Entity value* the
# two scorecards genuinely do run on different entities of one application --
# which a pipeline-level or deployment-level A/B could not express at all.
# --------------------------------------------------------------------------
