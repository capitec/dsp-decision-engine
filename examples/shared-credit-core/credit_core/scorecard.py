"""The retail unsecured scorecard: points per characteristic, with the reason code each low score raises."""
from __future__ import annotations

from decider import ScorecardConfig

SCORECARD_ID = 1
SCORECARD_VERSION = "sc-2026.09"

# characteristic -> the reason code raised when its points are negative.
_REASON_CODE_BY_VARIABLE = (
    ("bureau_score", 4101),
    ("months_employed", 4102),
    ("worst_arrears_months", 4103),
    ("accounts_in_arrears_count", 4104),
    ("revolving_utilisation", 4105),
    ("applicant_age_years", 4106),
    ("dependants_count", 4107),
    ("employment_type_code", 4108),
)
VARIABLES = tuple(name for name, _ in _REASON_CODE_BY_VARIABLE)


def _edge(name: str, default: float) -> dict:
    return {"param": name, "default": default}


def _bins(*edges_and_points: tuple[float | None, float | None, int]) -> list[dict]:
    """Bins from `(lower, upper, points)` triples; a missing edge is open."""
    return [{"value": points, **({"lower_bound": lo} if lo is not None else {}),
             **({"upper_bound": hi} if hi is not None else {})} for lo, hi, points in edges_and_points]


def _scored(variable: str, bins: list[dict], default: int) -> dict:
    return {"type": "scored", "variable_name": variable, "strict": False, "default": {"value": default}, "bins": bins}


def build_scorecard() -> ScorecardConfig:
    """Writes `score` and, per characteristic, `<characteristic>_score`.

    The bureau score's cut-offs are params (`bureau_low`, `bureau_mid`, `bureau_high`); a missing
    characteristic scores its default bin instead of failing.
    """
    low, mid, high = _edge("bureau_low", 500.0), _edge("bureau_mid", 650.0), _edge("bureau_high", 750.0)
    return ScorecardConfig.load({
        "type": "scorecard", "name": "retail_unsecured", "output_name": "score",
        "variables": [
            {"type": "constant", "score": 600, "output_name": "base_score"},  # puts totals in calibration's range
            _scored("bureau_score", _bins((None, low, -40), (low, mid, -10), (mid, high, 20), (high, None, 45)), 0),
            _scored("months_employed", _bins((None, 6.0, -15), (6.0, 36.0, 0), (36.0, None, 20)), -5),
            _scored("worst_arrears_months", _bins((None, 0.5, 25), (0.5, 2.0, 0), (2.0, None, -35)), 0),
            _scored("accounts_in_arrears_count",
                    [{"value": 15, "items": [0]}, {"value": -10, "items": [1]}, {"value": -25, "items": [2]}], -40),
            _scored("revolving_utilisation", _bins((None, 0.3, 10), (0.3, 0.7, -5), (0.7, None, -20)), 0),
            _scored("applicant_age_years", _bins((None, 25.0, -5), (25.0, 60.0, 5), (60.0, None, 0)), 0),
            _scored("dependants_count", _bins((None, 2.5, 5), (2.5, None, -5)), 0),
            _scored("employment_type_code",
                    [{"value": 10, "items": [1]}, {"value": 0, "items": [2, 3]}, {"value": -15, "items": [4, 5, 6]}], -10),
        ],
    })


def adverse_action_codes(scores: dict[str, float], n: int = 4) -> list[int]:
    """Reason codes of the `n` most negative characteristics, worst first.

    `scores` maps characteristic to points, e.g. `{"bureau_score": -40.0, "months_employed": 20.0}`.
    """
    negative = sorted(((points, name) for name, points in scores.items() if points < 0))
    codes = dict(_REASON_CODE_BY_VARIABLE)
    return [codes[name] for _, name in negative[:n]]
