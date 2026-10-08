"""Segment, score, probability of default and risk grade, with four approved overlays on the model's answers.

The scorecard, the calibration and the grade boundaries come from `credit_core` untouched. Policy adjusts what they
say through overlays, in a declared order, because calibration sits between them: a score shift, then a change to the
scaling, then an odds multiplier on the probability of default, then a boundary shift. Every overlay keeps the
unadjusted value beside the adjusted one, and `adjustment_stack_enabled` switches each point off through the same code.
"""
from __future__ import annotations

from datetime import date

from decider import dag, missing_as, step

from credit_core import calibration, risk_grade
from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.scorecard import VARIABLES, adverse_action_codes, build_scorecard
from credit_core.vocab import SEGMENT_NEW_TO_BANK, SEGMENT_RETAIL_MASS

SCORECARD_THIN_FILE, SCORECARD_NEW_TO_BANK, SCORECARD_EXISTING_CLIENT = 1012, 1010, 1011
ADJUSTMENT_SET_ID = "AS-03-2026.09"


@step(outputs=("scorecard_id", "segment_code"))
def segment_assignment(is_thin_file: bool, internal_tenure_months: float = missing_as(0.0)) -> tuple[int, float]:
    """Thin file first, then new to bank, then existing client: the scorecard that scores the application and its segment.

    Calibration and the grade boundaries are keyed by segment; thin-file and new-to-bank clients share the wider
    bands `credit_core` keeps for the new-to-bank segment.
    """
    if is_thin_file:
        return SCORECARD_THIN_FILE, SEGMENT_NEW_TO_BANK
    if internal_tenure_months > 0:
        return SCORECARD_EXISTING_CLIENT, SEGMENT_RETAIL_MASS
    return SCORECARD_NEW_TO_BANK, SEGMENT_NEW_TO_BANK


def score_reason_codes(
    bureau_score_score: float, months_employed_score: float, worst_arrears_months_score: float,
    accounts_in_arrears_count_score: float, revolving_utilisation_score: float, applicant_age_years_score: float,
    dependants_count_score: float, employment_type_code_score: float,
) -> list[int]:
    """The reason codes of the four characteristics that cost the most points; the adverse-action explanation."""
    points = dict(zip(VARIABLES, (bureau_score_score, months_employed_score, worst_arrears_months_score,
                                  accounts_in_arrears_count_score, revolving_utilisation_score,
                                  applicant_age_years_score, dependants_count_score, employment_type_code_score)))
    return adverse_action_codes(points, n=4)


SCORE_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-03-001", kind="score_shift", target="score", effect=AdjustmentEffect("add", -18.0),
        scope={"scorecard_id": SCORECARD_THIN_FILE}, stack_position=1, owner="Model Risk",
        approval_reference="MRC-2026-014", rationale="Thin-file defaults running above prediction for two quarters",
        effective_from=date(2026, 1, 1), effective_to=date(2027, 3, 31), review_date=date(2027, 1, 31),
        tighten_only=True),
    Adjustment(
        adjustment_id="ADJ-03-002", kind="scaling_change", target="calibration_scale",
        effect=AdjustmentEffect("multiply", 1.10), scope={"scorecard_id": SCORECARD_EXISTING_CLIENT},
        stack_position=2, owner="Model Risk", approval_reference="MRC-2026-015",
        rationale="Points to double the odds widened on the existing-client segment, flattening the tails",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31)),
    Adjustment(
        adjustment_id="ADJ-03-003", kind="odds_multiplier", target="probability_of_default",
        effect=AdjustmentEffect("multiply", 1.35), scope={"channel_code": 4}, stack_position=3,
        owner="Credit Risk Policy", approval_reference="CRC-2026-041",
        rationale="Channel 4 realised defaults running above the model's prediction",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True),
    # workaround: credit_core has no boundary-shift overlay, so moving the grade 6/7 boundary down is expressed as
    # an odds multiplier on the PD that is graded; the risk grade table itself is never edited.
    Adjustment(
        adjustment_id="ADJ-03-004", kind="odds_multiplier", target="probability_of_default_for_grading",
        effect=AdjustmentEffect("multiply", 1.12), scope={"product_code": 10}, stack_position=4,
        owner="Credit Risk Policy", approval_reference="CRC-2026-042",
        rationale="The top of the acceptable risk range tightened",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True),
])


def _overlay(target: str, base_field: str):
    """One overlay point. Each writes its own `<target>_adjustment_set_id` and `<target>_adjustments_applied`."""
    return SCORE_ADJUSTMENTS.apply_stack_step(target, ADJUSTMENT_SET_ID, base_field=base_field).relabel(
        writes={"adjustment_set_id": f"{target}_adjustment_set_id",
                "adjustments_applied": f"{target}_adjustments_applied"})


def adjustments_applied(
    score_adjustments_applied: list[str], calibration_scale_adjustments_applied: list[str],
    probability_of_default_adjustments_applied: list[str],
    probability_of_default_for_grading_adjustments_applied: list[str],
) -> list[str]:
    """Every overlay that moved the result, in the order the stack applied them."""
    return [*score_adjustments_applied, *calibration_scale_adjustments_applied,
            *probability_of_default_adjustments_applied, *probability_of_default_for_grading_adjustments_applied]


def adjustment_set_id(score_adjustment_set_id: str) -> str:
    return score_adjustment_set_id


def unit():
    """Segment, score, overlays 1 to 4 and the grade. Reads the bureau characteristics and writes `risk_grade`."""
    return dag(
        segment_assignment,
        build_scorecard().relabel(writes={"score": "score_raw"}, reads={"dependants_count": "applicant1_dependants_count"}),
        score_reason_codes,
        _overlay("score", "score_raw"),
        calibration.build_calibration_table(),
        _overlay("calibration_scale", "scale"),
        step(calibration.probability_of_default, output="probability_of_default_raw").relabel(
            reads={"scale": "calibration_scale"}),
        _overlay("probability_of_default", "probability_of_default_raw"),
        _overlay("probability_of_default_for_grading", "probability_of_default"),
        risk_grade.build_risk_grade_table().relabel(reads={"probability_of_default": "probability_of_default_for_grading"}),
        step(adjustments_applied), step(adjustment_set_id),
        name="scoring",
    ).emit(
        "scorecard_id", "segment_code", "score", "score_unadjusted", "score_reason_codes", "probability_of_default",
        "probability_of_default_unadjusted", "risk_grade", "risk_grade_cell_id", "risk_grade_version",
        "risk_grade_boundary_lo", "risk_grade_boundary_hi", "calibration_cell_id", "calibration_version",
        "adjustments_applied", "adjustment_set_id")
