"""Credit life premium: a rate per R1 000 of cover by age, term and cover type, capped by statute."""
from __future__ import annotations

from decider import DecisionTableConfig, param, step

from credit_core.evidence import cell_id

CREDIT_LIFE_VERSION = "cl-2026.09"
SINGLE = 1
JOINT = 2

_INF = float("inf")
_AGE_BANDS = [(-_INF, 30.0), (30.0, 55.0), (55.0, _INF)]
_TERM_BANDS = [(-_INF, 24.0), (24.0, 60.0), (60.0, _INF)]
# Rate per R1 000 by cover type, then age band, then term band.
_RATES = {SINGLE: [[3.20, 3.80, 4.50], [4.00, 4.70, 5.60], [6.50, 7.80, 9.20]],
          JOINT: [[5.40, 6.40, 7.60], [6.70, 7.90, 9.40], [10.90, 13.10, 15.40]]}


def build_credit_life_table() -> DecisionTableConfig:
    """Writes `rate` (per R1 000 of cover) and `cell_id` for `applicant_age_years`, `term_months` and `cover_type_code`."""
    cells = [(cover, ai, ti) for cover in _RATES for ai in range(3) for ti in range(3)]
    rows = [{"age_lo": _AGE_BANDS[ai][0], "age_hi": _AGE_BANDS[ai][1], "term_lo": _TERM_BANDS[ti][0],
             "term_hi": _TERM_BANDS[ti][1], "cover_type": cover, "rate": _RATES[cover][ai][ti],
             "cell_id": cell_id("credit_life", CREDIT_LIFE_VERSION, i)} for i, (cover, ai, ti) in enumerate(cells)]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "credit_life_rates",
        "columns": {"age_lo": "Float64", "age_hi": "Float64", "term_lo": "Float64", "term_hi": "Float64",
                    "cover_type": "Int64", "rate": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "between", "variable": "applicant_age_years", "lower_bound_column": "age_lo",
             "upper_bound_column": "age_hi", "allow_gaps": True},
            {"type": "between", "variable": "term_months", "lower_bound_column": "term_lo",
             "upper_bound_column": "term_hi", "allow_gaps": True},
            {"type": "eq", "variable": "cover_type_code", "value_column": "cover_type"},
        ]},
        "outputs": ["rate", "cell_id"],
        "default": [0.0, None],
    })


def credit_life_premium(offered_amount: float, rate: float, cap: float = param(350.0, ge=0.0)) -> float:
    return round(min(offered_amount / 1000.0 * rate, cap), 2)


def credit_life_cap_applied(offered_amount: float, rate: float, cap: float = param(350.0, ge=0.0)) -> bool:
    return offered_amount / 1000.0 * rate >= cap


credit_life_premium_step = step(credit_life_premium)
credit_life_cap_applied_step = step(credit_life_cap_applied)
