"""Stage 5.8, PP-05 and PP-11 -- the blend, and its three required outputs.

PP-05: "The blend is over probabilities of default on the log-odds scale, not
over grades. Averaging grade numbers is arithmetically wrong because grades are
not linear in risk, and doing it was a real defect in the previous generation."

The interesting part of this file is not the blend -- it is that the blend is
computed **three times over the same weights** and that doing so costs one
declaration rather than three implementations.
"""

from decider2 import module, param, step, Gather, sum_of, count, Shadow
from grains import Entity, Application


# --------------------------------------------------------------------------
# Entity grain. The clever bit is a step; the fold is a plain sum. That is the
# division of labour `Gather` exists to enforce: folds stay commutative and
# stupid, arithmetic stays in testable scalar steps.
# --------------------------------------------------------------------------
def entity_log_odds(probability_of_default: float) -> float:
    """log(pd / (1 - pd)). Guarded at both ends by the calibration's own floor
    and ceiling, because a PD of exactly 0 or 1 is a calibration defect and must
    surface as one rather than as an infinity in a sum."""
    pass


def entity_log_odds_unadjusted(probability_of_default_unadjusted: float) -> float:
    """The same transform over the pre-overlay PD. PP-11 requires the unadjusted
    blend to use IDENTICAL weights, so both quantities are weighted at the Entity
    grain and summed by the same fold declaration."""
    pass


def weighted_log_odds(entity_log_odds: float, blend_weight: float) -> float:
    """Contribution of this entity to the people PD."""
    pass


def weighted_log_odds_unadjusted(
    entity_log_odds_unadjusted: float, blend_weight: float
) -> float:
    """Contribution using the pre-overlay PD at the same weight."""
    pass


def entity_overlay_log_odds_delta(
    entity_log_odds: float, entity_log_odds_unadjusted: float
) -> float:
    """How far this entity's own overlays moved its log-odds."""
    pass


def weighted_overlay_delta(
    entity_overlay_log_odds_delta: float, blend_weight: float
) -> float:
    """PP-11's decomposition term: this entity's overlay effect at its weight.

    Summing these gives the total overlay contribution; keeping them per entity
    gives PP-11's required decomposition "by entity and by overlay". The
    per-overlay half comes from `adjustments_applied`, which the scorecard
    emitted per entity in s5.7.
    """
    pass


BlendTerms = module(
    entity_log_odds, entity_log_odds_unadjusted,
    weighted_log_odds, weighted_log_odds_unadjusted,
    entity_overlay_log_odds_delta, weighted_overlay_delta,
    name="blend_terms", grain=Entity,
    taps=["blend_weight", "weighted_log_odds", "weighted_overlay_delta"],
)


# --------------------------------------------------------------------------
# One Gather, three sums. This is PP-11's three required outputs:
#
#   people_pd                       the blend of adjusted entity PDs
#   people_pd_unadjusted            the same blend over unadjusted PDs,
#                                   with IDENTICAL weights
#   people_pd_overlay_contribution  the difference, decomposed by entity
#
# The identical-weights requirement is satisfied structurally: both sums read
# `blend_weight`, which is one value computed once. An implementation that
# re-derived the weights for the unadjusted run could get them subtly different
# -- and PP-11 exists because the Bank has to be able to distinguish "the
# entity's data changed" from "the model changed" from "an overlay changed",
# which is impossible if the weights moved too.
# --------------------------------------------------------------------------
PeopleBlendFacts = Gather(
    Entity, into=Application, name="people_blend_facts",
    people_log_odds            = sum_of("weighted_log_odds", where="in_blend"),
    people_log_odds_unadjusted = sum_of("weighted_log_odds_unadjusted", where="in_blend"),
    overlay_log_odds_delta     = sum_of("weighted_overlay_delta", where="in_blend"),
    blend_entity_count         = count(where="in_blend"),
    weight_sum_check           = sum_of("blend_weight", where="in_blend"),
)


def people_pd(people_log_odds: float) -> float:
    """Inverse log-odds. The operative figure."""
    pass


def people_pd_unadjusted(people_log_odds_unadjusted: float) -> float:
    """What the models alone said."""
    pass


def people_pd_overlay_contribution(
    people_pd: float, people_pd_unadjusted: float
) -> float:
    """The difference. The per-entity decomposition travels separately, as an
    Entity-grain tap, because s5.8 requires it "by entity and by overlay"."""
    pass


def people_grade(people_pd: float, segment_code: int) -> int:
    """core.risk_grade on the `sme-people` segment. NOT an average of grades."""
    pass


def people_grade_unadjusted(people_pd_unadjusted: float, segment_code: int) -> int:
    """The grade the models alone would have given."""
    pass


def weights_normalise(
    weight_sum_check: float,
    tolerance: float = param(1e-9, ge=0.0),
) -> bool:
    """A consistency assertion, not a decision: the weight vector must sum to 1.

    It is here rather than in validation/consistency.py because a weight vector
    that does not sum to 1 makes both blends wrong in the same direction, which
    is the failure that looks like a model drift and is not one.
    """
    pass


PeopleBlend = (
    BlendTerms
    | PeopleBlendFacts
    | module(people_pd, people_pd_unadjusted, people_pd_overlay_contribution,
             people_grade, people_grade_unadjusted, weights_normalise,
             name="people_blend", grain=Application,
             taps=["people_pd", "people_pd_unadjusted", "people_grade",
                   "people_grade_unadjusted", "people_pd_overlay_contribution"],
             contract="contracts/people_blend.json")
)
