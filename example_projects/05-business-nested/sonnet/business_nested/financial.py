"""Stage 7 -- business financial assessment (spec 05 §5.9), three ratios (SCOPE.md
explicitly caps this slice at 3 of the spec's eleven derived measures): interest
cover, current ratio and gearing. The 48-line spread, sector benchmarking and the
audit-level haircut table are out of this slice.

Reuses `decider`'s `ScorecardConfig` (a second, small one -- the same built-in
`scoring.py`'s entity scorecard uses) and project 00's `core.calibration`,
keyed on `SEGMENT_SME_ENTITY` -- a third "same capability, different settings"
instance in this project (entity scoring, the people blend's risk-grade lookup,
and this)."""
from __future__ import annotations

from decider import dag, missing_as, param, step
from decider.steps.scorecard import ScorecardConfig

from credit_core.calibration import build_calibration_table, probability_of_default_step

from business_nested import vocab
from business_nested.scoring import grade_from_pd

FINANCIAL_SCORECARD_VERSION = "fin-sc-2026.09"


def interest_cover(ebitda: float, finance_charges: float = missing_as(0.0)) -> float:
    return ebitda / finance_charges if finance_charges > 0 else float("inf") if ebitda >= 0 else 0.0


def current_ratio(current_assets: float, current_liabilities: float = missing_as(0.0)) -> float:
    return current_assets / current_liabilities if current_liabilities > 0 else float("inf")


def gearing(interest_bearing_debt: float, tangible_net_worth: float = missing_as(1.0)) -> float:
    return interest_bearing_debt / tangible_net_worth if tangible_net_worth > 0 else 99.0


def financial_confidence_code(
    finance_charges: float = missing_as(0.0), current_liabilities: float = missing_as(0.0),
    tangible_net_worth: float = missing_as(0.0),
) -> str:
    """spec 05 §5.9's four-level confidence, reduced to two -- every ratio computable
    cleanly (a real denominator on all three) is "high"; any ratio that had to fall
    back to its guard value is "low". Medium/none (statement-based, bank-turnover
    fallback) are out of this slice's 3-ratio scope."""
    return "high" if finance_charges > 0 and current_liabilities > 0 and tangible_net_worth > 0 else "low"


def build_financial_scorecard() -> ScorecardConfig:
    return ScorecardConfig.load({
        "type": "scorecard", "name": "financial_scorecard", "output_name": "financial_score_raw",
        "variables": [
            {"type": "constant", "score": 600, "output_name": "financial_scorecard_offset"},
            {"type": "scored", "variable_name": "interest_cover", "strict": False,
             "default": {"value": -20, "name": "unknown"}, "bins": [
                 {"value": -30, "upper_bound": 1.5, "name": "weak"},
                 {"value": 0, "lower_bound": 1.5, "upper_bound": 3.0, "name": "adequate"},
                 {"value": 30, "lower_bound": 3.0, "name": "strong"},
             ]},
            {"type": "scored", "variable_name": "current_ratio", "strict": False,
             "default": {"value": -10, "name": "unknown"}, "bins": [
                 {"value": -20, "upper_bound": 1.0, "name": "weak"},
                 {"value": 0, "lower_bound": 1.0, "upper_bound": 1.5, "name": "adequate"},
                 {"value": 15, "lower_bound": 1.5, "name": "strong"},
             ]},
            {"type": "scored", "variable_name": "gearing", "strict": False,
             "default": {"value": -10, "name": "unknown"}, "bins": [
                 {"value": 15, "upper_bound": 1.0, "name": "low"},
                 {"value": 0, "lower_bound": 1.0, "upper_bound": 2.5, "name": "moderate"},
                 {"value": -25, "lower_bound": 2.5, "name": "high"},
             ]},
        ],
    })


def financial_segment_code() -> int:
    return vocab.SEGMENT_SME_ENTITY


def financial_grade_output(financial_pd: float) -> int:
    return grade_from_pd(financial_pd, vocab.SEGMENT_SME_ENTITY)


def build_financial_unit():
    segment_step = step(financial_segment_code, output="segment_code")
    pd_step = probability_of_default_step.relabel(
        reads={"score": "financial_score_raw"}, writes={"probability_of_default": "financial_pd"})
    return dag(
        step(interest_cover), step(current_ratio), step(gearing), step(financial_confidence_code),
        segment_step, build_financial_scorecard(), build_calibration_table(), pd_step,
        step(financial_grade_output, output="financial_grade"),
        name="financial",
    ).emit(
        "interest_cover", "current_ratio", "gearing", "financial_confidence_code", "financial_score_raw",
        "financial_pd", "financial_grade",
    )
