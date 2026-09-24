"""Stage 5.4 -- application scorecard, calibration, grading, and the four
post-model overlays (spec 03 §5.4, §5.4.1).

Segment precedence (thin-file > new-to-bank > existing-client) is real
logic, computed for evidence and for `scorecard_id`/`segment_code`
attribution, exactly as §5.4 describes. SCOPE.md reduces the *scorecard
count* to one for this slice ("one scorecard with calibration and grading
plus the §5.4.1 overlays"), so every segment scores through
`credit_core.scorecard.build_scorecard()` (project 00's single reused
scorecard) rather than through four segment-specific cards -- the segment
assignment still determines which `segment_code` calibration and grading
key on, which is the part 06/other consumers actually depend on.

The four overlays compose in the declared order §5.4.1 requires (score,
then scaling, then odds, then boundary), each layered with
`credit_core.adjustments.AdjustmentRegister` (project 00's mechanism,
unmodified) so the unadjusted value survives at every step and the whole
stack can run disabled through the same code path.

**Boundary shift, implemented as an equivalent PD-for-grading multiplier.**
Project 00's `_TIGHTEN_RULES` registry (`credit_core/adjustments.py`) is
closed over five kinds and cannot be extended without editing project 00
(never done -- see BRIEF). "Moving the grade 6/7 boundary from 4.60% to
4.20%" and "multiplying the PD fed to the (unmoved) boundary table by a
tighten-only factor >= 1.0" select the same grade for the same underlying
risk, for a monotone step function -- so the boundary-shift overlay reuses
`kind="odds_multiplier"`'s tighten rule (already declared: `multiply, value
>= 1.0`) on a *different* target, `probability_of_default_for_grading`,
rather than `probability_of_default` itself. The risk_grade table (and its
boundaries) is never edited; only the value tested against it moves -- the
same property §5.4.1 requires of a real boundary shift. Recorded in
NOTES.md "Framework friction".
"""
from __future__ import annotations

from datetime import date

from decider import dag, flow, missing_as, param, step
from decider.steps.tables import DecisionTableConfig

from credit_core import calibration, risk_grade as rg
from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.scorecard import VARIABLES, adverse_action_codes, build_scorecard
from credit_core.vocab import SEGMENT_NEW_TO_BANK, SEGMENT_RETAIL_MASS

SCORECARD_THIN_FILE = 1012
SCORECARD_NEW_TO_BANK = 1010
SCORECARD_EXISTING_CLIENT = 1011

# segment_code, for calibration/risk_grade table keys (00's tables carry (10,1),
# (10,2), (10,3) at working depth -- see NOTES.md "Gaps in what I consumed" for
# why thin-file and new-to-bank both map onto segment 3, the one 00 built with
# "wider bands" for a thinner-evidence population).
_SEGMENT_BY_SCORECARD = {
    SCORECARD_THIN_FILE: SEGMENT_NEW_TO_BANK,
    SCORECARD_NEW_TO_BANK: SEGMENT_NEW_TO_BANK,
    SCORECARD_EXISTING_CLIENT: SEGMENT_RETAIL_MASS,
}


def segment_assignment(
    bureau_account_count: int = missing_as(0), bureau_history_months: float = missing_as(0.0),
    internal_tenure_months: float = missing_as(0.0),
) -> tuple[int, int]:
    """Priority 1 thin-file, 2 new-to-bank, 3 existing-client (spec 03 §5.4)."""
    if bureau_account_count < 3 or bureau_history_months < 12.0:
        scorecard_id = SCORECARD_THIN_FILE
    elif not internal_tenure_months or internal_tenure_months <= 0:
        scorecard_id = SCORECARD_NEW_TO_BANK
    else:
        scorecard_id = SCORECARD_EXISTING_CLIENT
    return scorecard_id, _SEGMENT_BY_SCORECARD[scorecard_id]


segment_assignment_step = step(segment_assignment, outputs=("scorecard_id", "segment_code"))


def score_reason_codes(**scores: float) -> list[int]:
    contributions = {v: scores[f"{v}_score"] for v in VARIABLES}
    return adverse_action_codes(contributions, n=4)


def _score_reason_codes(
    bureau_score_score: float, months_employed_score: float, worst_arrears_months_score: float,
    accounts_in_arrears_count_score: float, revolving_utilisation_score: float, applicant_age_years_score: float,
    dependants_count_score: float, employment_type_code_score: float,
) -> list[int]:
    return score_reason_codes(
        bureau_score_score=bureau_score_score, months_employed_score=months_employed_score,
        worst_arrears_months_score=worst_arrears_months_score,
        accounts_in_arrears_count_score=accounts_in_arrears_count_score,
        revolving_utilisation_score=revolving_utilisation_score,
        applicant_age_years_score=applicant_age_years_score,
        dependants_count_score=dependants_count_score, employment_type_code_score=employment_type_code_score,
    )


score_reason_codes_step = step(_score_reason_codes, output="score_reason_codes")


ADJUSTMENT_SET_ID = "AS-03-2026.09"

SCORE_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-03-001", kind="score_shift", target="score",
        effect=AdjustmentEffect("add", -18.0), scope={"scorecard_id": SCORECARD_THIN_FILE}, stack_position=1,
        owner="Model Risk", approval_reference="MRC-2026-014",
        rationale="Thin-file segment defaults running above prediction for two quarters",
        effective_from=date(2026, 1, 1), effective_to=date(2027, 3, 31), review_date=date(2027, 1, 31),
        tighten_only=True,
    ),
    Adjustment(
        adjustment_id="ADJ-03-002", kind="scaling_change", target="calibration_scale",
        effect=AdjustmentEffect("multiply", 1.10), scope={"scorecard_id": SCORECARD_EXISTING_CLIENT}, stack_position=2,
        owner="Model Risk", approval_reference="MRC-2026-015",
        rationale="Points-to-double-the-odds widened on the existing-client segment, flattening the tails",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=False,
    ),
    Adjustment(
        adjustment_id="ADJ-03-003", kind="odds_multiplier", target="probability_of_default",
        effect=AdjustmentEffect("multiply", 1.35), scope={"channel_code": 4}, stack_position=3,
        owner="Credit Risk Policy", approval_reference="CRC-2026-041",
        rationale="Channel 4 realised defaults running above the model's prediction",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True,
    ),
    Adjustment(
        adjustment_id="ADJ-03-004", kind="odds_multiplier", target="probability_of_default_for_grading",
        effect=AdjustmentEffect("multiply", 1.12), scope={"product_code": 10}, stack_position=4,
        owner="Credit Risk Policy", approval_reference="CRC-2026-042",
        rationale="The grade 6/7 boundary tightened; expressed as an equivalent PD-for-grading multiplier "
                   "(see this module's docstring) rather than an edit to the risk_grade table",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True,
    ),
])


def _overlay(target: str, base_field: str | None = None):
    """One overlay point, with its output columns relabelled `<target>_*` so composing
    four calls of `apply_stack_step` in one flow doesn't collide on `adjustment_set_id` /
    `adjustments_applied` (00's own column-naming convention, applied here per-overlay
    since 00's `cell_id` convention is per-table, not per-overlay-point)."""
    return SCORE_ADJUSTMENTS.apply_stack_step(target, ADJUSTMENT_SET_ID, base_field=base_field).relabel(
        writes={"adjustment_set_id": f"{target}_adjustment_set_id", "adjustments_applied": f"{target}_adjustments_applied"},
    )


def combined_adjustments_applied(
    score_adjustments_applied: list[str], calibration_scale_adjustments_applied: list[str],
    probability_of_default_adjustments_applied: list[str],
    probability_of_default_for_grading_adjustments_applied: list[str],
) -> list[str]:
    return (
        list(score_adjustments_applied) + list(calibration_scale_adjustments_applied)
        + list(probability_of_default_adjustments_applied) + list(probability_of_default_for_grading_adjustments_applied)
    )


combined_adjustments_applied_step = step(combined_adjustments_applied, output="adjustments_applied")


def _adjustment_set_id(score_adjustment_set_id: str) -> str:
    """All four overlay points share one `AdjustmentRegister`/`adjustment_set_id`; this
    just picks the one canonical column name (00's convention: each `apply_stack_step`
    call writes its own copy, since they're relabelled apart to avoid a dag collision)."""
    return score_adjustment_set_id


adjustment_set_id_step = step(_adjustment_set_id, output="adjustment_set_id")


def build_scoring_unit():
    """Stage 5.4: segment -> score -> overlay 1 -> calibrate (with overlay 2 on scale)
    -> PD -> overlay 3 -> grade (with overlay 4 on the graded PD)."""
    scorecard_step = build_scorecard().relabel(writes={"score": "score_raw"})
    calibration_table = calibration.build_calibration_table()
    risk_grade_table = rg.build_risk_grade_table().relabel(
        reads={"probability_of_default": "probability_of_default_for_grading"},
    )

    return dag(
        segment_assignment_step,
        scorecard_step,
        score_reason_codes_step,
        _overlay("score", base_field="score_raw"),

        calibration_table,
        _overlay("calibration_scale", base_field="scale"),
        step(calibration.probability_of_default, output="probability_of_default_raw").relabel(
            reads={"scale": "calibration_scale"},
        ),
        _overlay("probability_of_default", base_field="probability_of_default_raw"),

        _overlay("probability_of_default_for_grading", base_field="probability_of_default"),
        risk_grade_table,
        step(rg.risk_grade_output, output="risk_grade"),

        combined_adjustments_applied_step,
        adjustment_set_id_step,
        name="scoring",
    )
