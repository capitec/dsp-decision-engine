"""Behavioural scoring (s5.3): 32 characteristics, 24-month window, one
scorecard per product, plus the three overlay points Model Risk and Credit Risk
Policy actually use.

The scorecard is a `scorecard` kind (doc 08 s3.4): bins -> points is naturally
tabular, so it runs on a generic kernel and a recalibration is a table swap,
not a compile. The per-characteristic contributions are `evidence`, not taps:
they are the raw material for the reason given to a client, so they may not be
dropped by a consumer (FRAMEWORK-DEMANDS #9).
"""

from decider2 import Branch, module, param, scorecard, shadow, table
from decider2.credit import core, overlay_point

BehaviouralScorecard = scorecard(
    "clm.behavioural",
    keys=("scorecard_id",),
    characteristics=32,
    bins_per_characteristic=9,
    null_bin=True,          # "insufficient history" and "no bureau record" are BINS
    effective_dated=True,
    evidence=["characteristic_vector", "point_contributions", "null_bins_used"],
)

GradeBoundaries = table(
    "clm.grade_boundaries",
    keys=("product_code", "grade"),
    values={"pd_upper": float},
    dense=True,
    effective_dated=True,
)


def behaviour_score_raw(characteristic_vector, tables, params) -> float:
    """Sum of binned point contributions for the product's scorecard."""
    return tables.behavioural.score(characteristic_vector, params.scorecard_id)


def probability_of_default_raw(behaviour_score: float, tables, params) -> float:
    """core.calibration, score -> 12-month PD, on the calibration in force."""
    return core.calibration.pd(behaviour_score, params.calibration_id, tables)


def behaviour_grade(probability_of_default: float, product_code: int, tables) -> int:
    """1 (best) .. 12 (worst), from the effective-dated boundary table."""
    return tables.grade_boundaries.grade(product_code, probability_of_default)


# --- overlays --------------------------------------------------------------
# s5.3: a score shift, a PD multiplier and a grade-boundary shift. Because
# `behaviour_grade` keys the matrix, a one-notch shift is not cosmetic -- it
# changes the limit offered. So the overlay point sits INSIDE the graph, before
# the grade, and renders in the pipeline diagram where a reviewer will see it.

ApplyScoreOverlays = overlay_point(
    "score_overlays",
    register="overlay_set",
    adjusts={"behaviour_score": "shift",
             "probability_of_default": "multiply"},
    scope_keys=["product_code", "channel_code", "origination_partner_code",
                "behaviour_grade", "mob_band"],
    order="declared",
    writes={"applied": "adjustments_applied", "set": "adjustment_set_id"},
    evidence=["adjustments_applied", "adjustment_set_id"],
)

ApplyGradeBoundaryOverlays = overlay_point(
    "grade_boundary_overlays",
    register="overlay_set",
    adjusts={"grade_boundaries": "boundary_shift"},   # adjusts a TABLE, not a value
    scope_keys=["product_code"],
    order="declared",
)

Score = module(behaviour_score_raw, name="score_raw")
Calibrate = module(probability_of_default_raw, name="calibrate")
Grade = module(behaviour_grade, name="grade")

# The unadjusted score, PD and grade are not a second calculation. `shadow`
# re-emits the same sub-graph with the overlay register neutralised, so the two
# cannot drift -- there is one authored expression of the scorecard.
Scoring = (
    Score
    | Calibrate
    | ApplyScoreOverlays
    | ApplyGradeBoundaryOverlays
    | Grade
) | shadow(
    Score | Calibrate | Grade,
    neutralise={"overlay_set": "off"},
    suffix="_unadjusted",
    keep=["behaviour_score", "probability_of_default", "behaviour_grade"],
)

# Change scenario 4: two card scorecards live at once for a three-month
# parallel run on a 10% holdout, with a PD multiplier on the new one. One line,
# and the holdout flag is an ordinary input column.
ScoringWithHoldout = Branch(
    lambda in_holdout: 1 if in_holdout else 0,
    [Scoring, Scoring.rebind(scorecard_id=1104, calibration_id=1104)],
    modifies=["behaviour_score", "probability_of_default", "behaviour_grade"],
    name="scorecard_parallel_run",
)
