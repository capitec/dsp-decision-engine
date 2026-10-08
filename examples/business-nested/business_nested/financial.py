"""The business's own financial strength: three ratios, a scorecard, a probability of default and a grade."""
from __future__ import annotations

from decider import ScorecardConfig, flow, missing_as, step

from credit_core import calibration
from credit_core.vocab import SEGMENT_SME_ENTITY

from business_nested import scoring

RATIO_CAP = 99.0  # a ratio with no denominator is as good, or as bad, as it gets

CONFIDENCE_HIGH = 1  # every ratio has a real denominator
CONFIDENCE_LOW = 2


def interest_cover(ebitda: float, finance_charges: float = missing_as(0.0)) -> float:
    """Earnings per rand of finance charges; fully covered when there are none and earnings are not negative."""
    if finance_charges > 0:
        return ebitda / finance_charges
    return RATIO_CAP if ebitda >= 0 else 0.0


def current_ratio(current_assets: float, current_liabilities: float = missing_as(0.0)) -> float:
    return current_assets / current_liabilities if current_liabilities > 0 else RATIO_CAP


def gearing(interest_bearing_debt: float, tangible_net_worth: float = missing_as(0.0)) -> float:
    """Debt per rand of tangible net worth; capped when there is no positive net worth."""
    return interest_bearing_debt / tangible_net_worth if tangible_net_worth > 0 else RATIO_CAP


def financial_confidence_code(
    finance_charges: float = missing_as(0.0), current_liabilities: float = missing_as(0.0),
    tangible_net_worth: float = missing_as(0.0),
) -> int:
    return CONFIDENCE_HIGH if finance_charges > 0 and current_liabilities > 0 and tangible_net_worth > 0 else CONFIDENCE_LOW


def segment_code() -> float:
    return float(SEGMENT_SME_ENTITY)


def build_financial_scorecard() -> ScorecardConfig:
    """Writes `financial_score`; a missing ratio scores its default bin."""
    return ScorecardConfig.load({
        "type": "scorecard", "name": "financial_scorecard", "output_name": "financial_score",
        "variables": [
            {"type": "constant", "score": 600, "output_name": "financial_scorecard_offset"},
            {"type": "scored", "variable_name": "interest_cover", "strict": False, "default": {"value": -20},
             "bins": [{"value": -30, "upper_bound": 1.5}, {"value": 0, "lower_bound": 1.5, "upper_bound": 3.0},
                      {"value": 30, "lower_bound": 3.0}]},
            {"type": "scored", "variable_name": "current_ratio", "strict": False, "default": {"value": -10},
             "bins": [{"value": -20, "upper_bound": 1.0}, {"value": 0, "lower_bound": 1.0, "upper_bound": 1.5},
                      {"value": 15, "lower_bound": 1.5}]},
            {"type": "scored", "variable_name": "gearing", "strict": False, "default": {"value": -10},
             "bins": [{"value": 15, "upper_bound": 1.0}, {"value": 0, "lower_bound": 1.0, "upper_bound": 2.5},
                      {"value": -25, "lower_bound": 2.5}]},
        ],
    })


def financial_unit():
    """Ratios from the statements in the request to `financial_pd` and `financial_grade`."""
    return flow(
        step(interest_cover), step(current_ratio), step(gearing), step(financial_confidence_code),
        step(segment_code, output="segment_code"), build_financial_scorecard(), calibration.build_calibration_table(),
        calibration.probability_of_default_step.relabel(
            reads={"score": "financial_score"}, writes={"probability_of_default": "financial_pd"}),
        scoring.grade_lookup("financial_pd", "financial_grade"),
        name="financial",
    ).emit("interest_cover", "current_ratio", "gearing", "financial_confidence_code", "financial_score", "financial_pd",
           "financial_grade")
