"""`core.scorecard` -- scorecard evaluation (spec 00 §6.10; addendum A6).

Per-characteristic points are a required output, not a debugging nicety:
adverse-action explanation depends on them (§6.10 "Hard part"). Built on
`decider.steps.scorecard.ScorecardConfig` directly rather than
reimplemented, since binning-to-points is exactly what it does; nulls bin
to the declared default rather than erroring (a null is a scoring bin, not
an error -- §6.10).

# ponytail: 8 characteristics, not the spec's 20-60. SCOPE.md keeps only
# `core.rate_card` at its full declared dimension for this slice; every
# other table is "working depth". Add characteristics by appending to
# `VARIABLES` and `_score_columns()` below -- the mechanism doesn't change.

Addendum A6 (more than one score per decision): `scorecard_id` and
`scorecard_version` travel with every score so consumers can tell scores
apart by scorecard and role, instead of the library assuming one `score`
per decision.
"""
from __future__ import annotations

from decider.steps.scorecard import ScorecardConfig

SCORECARD_ID = 1
SCORECARD_VERSION = "sc-2026.09"

# Characteristic -> (reason code fired when this characteristic scores low).
# A small, fixed set stands in for the real 20-60 characteristic vector;
# each is still a genuine bin-to-points variable, not a stub.
_REASON_CODE_BY_VARIABLE = {
    "bureau_score": 4101,
    "months_employed": 4102,
    "worst_arrears_months": 4103,
    "accounts_in_arrears_count": 4104,
    "revolving_utilisation": 4105,
    "applicant_age_years": 4106,
    "dependants_count": 4107,
    "employment_type_code": 4108,
}
VARIABLES = tuple(_REASON_CODE_BY_VARIABLE)


def build_scorecard() -> ScorecardConfig:
    """The retail unsecured scorecard: bureau, tenure, arrears, utilisation, demographics.

    Bin edges are `param()`s (via the JSON `{"param": ..., "default": ...}`
    spelling), so Credit Risk Policy retunes a cut-off without a rebuild.
    """
    return ScorecardConfig.load({
        "type": "scorecard", "name": "retail_unsecured",
        "output_name": "score",
        "variables": [
            # A base offset, like every real points scorecard has, so the summed score lands
            # in the ~500-700 range `core.calibration`'s anchor/scale (calibration.py) expects,
            # rather than the characteristics' raw point spread (roughly -50..+100) alone.
            {"type": "constant", "score": 600, "output_name": "base_score"},
            {"type": "scored", "variable_name": "bureau_score", "strict": False, "default": {"value": 0, "name": "no_bureau"},
             "bins": [
                 {"value": -40, "upper_bound": {"param": "bureau_low", "default": 500.0}, "name": "very_low"},
                 {"value": -10, "lower_bound": {"param": "bureau_low", "default": 500.0},
                  "upper_bound": {"param": "bureau_mid", "default": 650.0}, "name": "low"},
                 {"value": 20, "lower_bound": {"param": "bureau_mid", "default": 650.0},
                  "upper_bound": {"param": "bureau_high", "default": 750.0}, "name": "good"},
                 {"value": 45, "lower_bound": {"param": "bureau_high", "default": 750.0}, "name": "excellent"},
             ]},
            {"type": "scored", "variable_name": "months_employed", "strict": False, "default": {"value": -5, "name": "unknown"},
             "bins": [
                 {"value": -15, "upper_bound": 6.0, "name": "under_6m"},
                 {"value": 0, "lower_bound": 6.0, "upper_bound": 36.0, "name": "6_to_36m"},
                 {"value": 20, "lower_bound": 36.0, "name": "over_36m"},
             ]},
            {"type": "scored", "variable_name": "worst_arrears_months", "strict": False, "default": {"value": 0, "name": "no_history"},
             "bins": [
                 {"value": 25, "upper_bound": 0.5, "name": "never"},
                 {"value": 0, "lower_bound": 0.5, "upper_bound": 2.0, "name": "minor"},
                 {"value": -35, "lower_bound": 2.0, "name": "material"},
             ]},
            {"type": "scored", "variable_name": "accounts_in_arrears_count", "strict": False, "default": {"value": -40, "name": "3_or_more"},
             "bins": [
                 {"value": 15, "items": [0]},
                 {"value": -10, "items": [1]},
                 {"value": -25, "items": [2]},
             ]},
            {"type": "scored", "variable_name": "revolving_utilisation", "strict": False, "default": {"value": 0, "name": "no_revolving"},
             "bins": [
                 {"value": 10, "upper_bound": 0.3, "name": "low"},
                 {"value": -5, "lower_bound": 0.3, "upper_bound": 0.7, "name": "mid"},
                 {"value": -20, "lower_bound": 0.7, "name": "high"},
             ]},
            {"type": "scored", "variable_name": "applicant_age_years", "strict": False, "default": {"value": 0, "name": "unknown"},
             "bins": [
                 {"value": -5, "upper_bound": 25.0, "name": "young"},
                 {"value": 5, "lower_bound": 25.0, "upper_bound": 60.0, "name": "mid"},
                 {"value": 0, "lower_bound": 60.0, "name": "senior"},
             ]},
            {"type": "scored", "variable_name": "dependants_count", "strict": False, "default": {"value": 0, "name": "unknown"},
             "bins": [
                 {"value": 5, "upper_bound": 2.5, "name": "low"},
                 {"value": -5, "lower_bound": 2.5, "name": "high"},
             ]},
            {"type": "scored", "variable_name": "employment_type_code", "strict": False, "default": {"value": -10, "name": "unknown"},
             "bins": [
                 {"value": 10, "items": [1]},   # permanent
                 {"value": 0, "items": [2, 3]},  # contract, self-employed
                 {"value": -15, "items": [4, 5, 6]},  # pensioner, social grant, informal
             ]},
        ],
    })


def adverse_action_codes(scores: dict[str, float], n: int = 4) -> list[int]:
    """The reason codes for the `n` largest negative per-characteristic contributions (§6.10).

    `scores` is `{variable_name: points}` -- the scorecard's own per-variable
    output columns, read back after `run`/`score`. Not itself a step: it
    operates on the whole contribution set at once, most naturally called
    by the consuming project's evidence assembly rather than per-column.
    """
    negative = sorted(((v, pts) for v, pts in scores.items() if pts < 0), key=lambda item: item[1])
    return [_REASON_CODE_BY_VARIABLE[v] for v, _ in negative[:n]]
