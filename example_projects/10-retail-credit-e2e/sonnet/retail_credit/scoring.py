"""P07 -- Scoring (spec 10 §5.8): this project's own scorecard.

Nine scorecards selected by segment x product x entry point is the spec's
full statement (10 §5.8, "28 rules, not a lookup"); this project runs one
product (10) on one entry point (1), so selection collapses to two cases,
both declared here: SC-10-A3 for an existing/clean-file segment, and
SC-10-A1 (the thin-file variant, with the spec's declared **-45 point**
degraded-bureau shift, 10 §5.8's "Degradation") for a new-to-bank or
bureau-degraded client. This is `retail_credit`'s **own** scorecard --
built with `decider.steps.scorecard.ScorecardConfig` directly (the
built-in project 00 exercises too), not `credit_core.scorecard`'s demo
instance, per 10 §1.1: "the library supplies the capability, this project
supplies the calibration."
"""
from __future__ import annotations

from decider import missing_as, param, step
from decider.steps.scorecard import ScorecardConfig

from retail_credit.vocab import DEGRADED_BUREAU_DOWN

SCORECARD_A1_ID = "SC-10-A1"  # thin-file / degraded-bureau fallback
SCORECARD_A3_ID = "SC-10-A3"  # existing-client, full bureau evidence
SCORECARD_VERSION = "sc10-2026.09"

_DEGRADED_SHIFT = -45.0


def build_scorecard_a3() -> ScorecardConfig:
    """The primary scorecard for this project's slice: bureau score, tenure, arrears,
    utilisation, income evidence tier, demographics -- eight characteristics, working
    depth (project 00's own scorecard note applies here too: the mechanism is identical
    at 8 or at 61, only the row count differs).
    """
    return ScorecardConfig.load({
        "type": "scorecard", "name": "retail_10_sc_a3",
        "output_name": "score",
        "variables": [
            {"type": "constant", "score": 600, "output_name": "base_score"},
            {"type": "scored", "variable_name": "bureau_score", "strict": False,
             "default": {"value": 0, "name": "no_bureau"},
             "bins": [
                 {"value": -45, "upper_bound": 500.0, "name": "very_low"},
                 {"value": -12, "lower_bound": 500.0, "upper_bound": 650.0, "name": "low"},
                 {"value": 22, "lower_bound": 650.0, "upper_bound": 750.0, "name": "good"},
                 {"value": 48, "lower_bound": 750.0, "name": "excellent"},
             ]},
            {"type": "scored", "variable_name": "months_employed", "strict": False,
             "default": {"value": -5, "name": "unknown"},
             "bins": [
                 {"value": -15, "upper_bound": 6.0, "name": "under_6m"},
                 {"value": 0, "lower_bound": 6.0, "upper_bound": 36.0, "name": "6_to_36m"},
                 {"value": 18, "lower_bound": 36.0, "name": "over_36m"},
             ]},
            {"type": "scored", "variable_name": "worst_arrears_months", "strict": False,
             "default": {"value": 0, "name": "no_history"},
             "bins": [
                 {"value": 28, "upper_bound": 0.5, "name": "never"},
                 {"value": -5, "lower_bound": 0.5, "upper_bound": 2.0, "name": "minor"},
                 {"value": -40, "lower_bound": 2.0, "name": "material"},
             ]},
            {"type": "scored", "variable_name": "accounts_in_arrears_count", "strict": False,
             "default": {"value": -40, "name": "3_or_more"},
             "bins": [{"value": 15, "items": [0]}, {"value": -10, "items": [1]}, {"value": -25, "items": [2]}]},
            {"type": "scored", "variable_name": "revolving_utilisation", "strict": False,
             "default": {"value": 0, "name": "no_revolving"},
             "bins": [
                 {"value": 10, "upper_bound": 0.3, "name": "low"},
                 {"value": -6, "lower_bound": 0.3, "upper_bound": 0.7, "name": "mid"},
                 {"value": -22, "lower_bound": 0.7, "name": "high"},
             ]},
            {"type": "scored", "variable_name": "income_evidence_tier", "strict": False,
             "default": {"value": -20, "name": "unknown"},
             "bins": [
                 {"value": 20, "items": [1, 2]},
                 {"value": 5, "items": [3, 4]},
                 {"value": -25, "items": [5, 6, 7]},
             ]},
            {"type": "scored", "variable_name": "applicant_age_years", "strict": False,
             "default": {"value": 0, "name": "unknown"},
             "bins": [
                 {"value": -6, "upper_bound": 25.0, "name": "young"},
                 {"value": 6, "lower_bound": 25.0, "upper_bound": 60.0, "name": "mid"},
                 {"value": 0, "lower_bound": 60.0, "name": "senior"},
             ]},
            {"type": "scored", "variable_name": "dependants_count", "strict": False,
             "default": {"value": 0, "name": "unknown"},
             "bins": [{"value": 5, "upper_bound": 2.5, "name": "low"}, {"value": -6, "lower_bound": 2.5, "name": "high"}]},
        ],
    })


_REASON_CODE_BY_VARIABLE = {
    "bureau_score": 4301, "months_employed": 4302, "worst_arrears_months": 4303,
    "accounts_in_arrears_count": 4304, "revolving_utilisation": 4305,
    "income_evidence_tier": 4306, "applicant_age_years": 4307, "dependants_count": 4308,
}


def select_scorecard(
    segment_code: int, degraded_mode_code: int = missing_as(0),
    bureau_hit: bool = missing_as(True),
) -> tuple[str, str]:
    """(scorecard_id, scorecard_version). Bureau-degraded or thin-file clients fall back
    to SC-10-A1 (10 §5.8's degradation path); everyone else scores on SC-10-A3.

    # ponytail: this selection is computed and recorded, but `pipeline.py`'s wired path
    # always scores on SC-10-A3 (nine scorecards selected by a 28-rule precedence is
    # project 10's stated dominant difficulty only at full product/segment breadth,
    # which SCOPE.md scopes to product 10 alone). A second `ScorecardConfig` plus a
    # `branch()` on this selector is the upgrade path once a bureau-down or thin-file
    # request needs to actually score on the fallback card rather than only compute
    # what it would have selected (proven standalone in `tests/test_scoring.py`).
    """
    if degraded_mode_code == DEGRADED_BUREAU_DOWN or not bureau_hit or segment_code in (1, 2):
        return SCORECARD_A1_ID, SCORECARD_VERSION
    return SCORECARD_A3_ID, SCORECARD_VERSION


def apply_degraded_shift(scorecard_id: str, score: float, shift: float = param(_DEGRADED_SHIFT)) -> float:
    """The spec's declared punitive shift (10 §5.8) when a client is forced onto the
    thin-file scorecard by a bureau outage rather than by genuinely being new-to-bank.
    Applied only on the fallback card, and only as a declared shift -- never silently
    folded into the base scorecard's own points.
    """
    return score + shift if scorecard_id == SCORECARD_A1_ID else score


select_scorecard_step = step(select_scorecard, outputs=("scorecard_id", "scorecard_version"))
apply_degraded_shift_step = step(apply_degraded_shift, output="score")
