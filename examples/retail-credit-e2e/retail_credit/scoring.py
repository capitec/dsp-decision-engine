"""The application scorecard: points for eight characteristics on top of a base of 600."""
from __future__ import annotations

from decikit import ScorecardConfig


def build_scorecard() -> ScorecardConfig:
    """Writes `score` for `bureau_score`, `months_employed`, `worst_arrears_months`, `accounts_in_arrears_count`,
    `revolving_utilisation`, `income_evidence_tier`, `applicant_age_years` and `dependants_count`."""

    def scored(variable: str, default: tuple[int, str], bins: list[dict]) -> dict:
        return {"type": "scored", "variable_name": variable, "strict": False,
                "default": {"value": default[0], "name": default[1]}, "bins": bins}

    return ScorecardConfig.load({
        "type": "scorecard", "name": "application_scorecard", "output_name": "score",
        "variables": [
            {"type": "constant", "score": 600, "output_name": "base_score"},
            scored("bureau_score", (0, "no_bureau"), [
                {"value": -45, "upper_bound": 500.0}, {"value": -12, "lower_bound": 500.0, "upper_bound": 650.0},
                {"value": 22, "lower_bound": 650.0, "upper_bound": 750.0}, {"value": 48, "lower_bound": 750.0}]),
            scored("months_employed", (-5, "unknown"), [
                {"value": -15, "upper_bound": 6.0}, {"value": 0, "lower_bound": 6.0, "upper_bound": 36.0},
                {"value": 18, "lower_bound": 36.0}]),
            scored("worst_arrears_months", (0, "no_history"), [
                {"value": 28, "upper_bound": 0.5}, {"value": -5, "lower_bound": 0.5, "upper_bound": 2.0},
                {"value": -40, "lower_bound": 2.0}]),
            scored("accounts_in_arrears_count", (-40, "3_or_more"), [
                {"value": 15, "items": [0]}, {"value": -10, "items": [1]}, {"value": -25, "items": [2]}]),
            scored("revolving_utilisation", (0, "no_revolving"), [
                {"value": 10, "upper_bound": 0.3}, {"value": -6, "lower_bound": 0.3, "upper_bound": 0.7},
                {"value": -22, "lower_bound": 0.7}]),
            scored("income_evidence_tier", (-20, "unknown"), [
                {"value": 20, "items": [1, 2]}, {"value": 5, "items": [3, 4]}, {"value": -25, "items": [5, 6, 7]}]),
            scored("applicant_age_years", (0, "unknown"), [
                {"value": -6, "upper_bound": 25.0}, {"value": 6, "lower_bound": 25.0, "upper_bound": 60.0},
                {"value": 0, "lower_bound": 60.0}]),
            scored("dependants_count", (0, "unknown"), [
                {"value": 5, "upper_bound": 2.5}, {"value": -6, "lower_bound": 2.5}]),
        ],
    })
