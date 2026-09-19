"""Stage 5.4 (part 3) -- calibration and grading, and the two overlay points
that sit on them.

Calibration is ten lines of arithmetic and one subtlety: the INVERSE is
required (§5.4 -- "appetite logic asks what score would we need for this to be
worth writing at this price") and the two directions must agree to within 0.01
of a point.  So `invertible_pair` is a declared relationship, not two
independent functions that happen to be written as inverses of each other, and
the agreement is a framework-generated test over the declared corpus rather
than a test somebody remembers to write.  The same mechanism serves
`core.instalment`'s inverse, which the solve's bracket depends on.
See FRAMEWORK-DEMANDS #13.
"""

from __future__ import annotations

from decider2 import invertible_pair, lookup_table, module, param, step


@invertible_pair(inverse="score_for_pd", agree_to=0.01, corpus="tests/corpora/calibration.csv")
@step(output="probability_of_default")
def pd_from_score(
    score: float,
    points_to_double_odds: float = param(20.0, gt=0, owner="model_risk"),
    anchor_score: float = param(600.0, owner="model_risk"),
    anchor_odds: float = param(30.0, gt=0, owner="model_risk"),
) -> float:
    """Log-odds scaling: `points_to_double_odds` points double the good:bad odds,
    anchored so that a score of `anchor_score` is `anchor_odds`:1."""
    pass  # factor = pdo / ln 2 = 28.8539; offset = 501.84 at the declared anchor


@step(output="score_for_pd")
def score_for_pd(
    target_probability_of_default: float,
    points_to_double_odds: float = param(20.0, gt=0, owner="model_risk"),
    anchor_score: float = param(600.0, owner="model_risk"),
    anchor_odds: float = param(30.0, gt=0, owner="model_risk"),
) -> float:
    """The score corresponding to a target PD. Used by appetite logic."""
    pass


Calibrate = module(pd_from_score, score_for_pd, name="calibration",
                   taps=["probability_of_default", "points_to_double_odds"])

# ---------------------------------------------------------------------------
# Grading.  48 boundary values -- 4 segments x 12 grades -- semi-annual, owned
# by Credit Risk Policy.  A lookup table, so a boundary move is a values
# change: free, no compile, no release.
#
# The BOUNDARY_SHIFT overlay in the pipeline moves one boundary for a declared
# scope and a declared period, WITHOUT editing this table -- which is the whole
# point of §5.4.1's "the scorecard is never edited".
# ---------------------------------------------------------------------------

GRADE_BOUNDARIES = lookup_table(
    name="grade_boundaries",
    axes={"segment_code": "ordinal", "grade": "dense(1..12)"},
    values=["pd_from", "pd_to"],
    source="tables/grade_boundaries.csv",
    owner="credit_risk_policy",
    effective_dated=True,
)


@step(output="risk_grade")
def grade(probability_of_default: float, segment_code: int, tables) -> int:
    """Grade 1 (best) to 12 (worst), from the boundary set in force at `decision_date`."""
    pass  # tables.grade_boundaries.band(segment_code, probability_of_default)


Grade = module(grade, name="grading", taps=["risk_grade", "grade_boundaries.version"])
