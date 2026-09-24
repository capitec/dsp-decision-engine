"""`core.credit_life` -- credit life premium (spec 00 §6.8). Thin, frozen interface.

Premium rate per R1 000 of cover, by age band x term band x employment
type (00 §8: 14 x 8 x 6 in full; working depth here: 3 age bands x 3 term
bands x 2 cover types), capped by statute.
"""
from __future__ import annotations

from decider import param, step
from decider.steps.tables import DecisionTableConfig

from credit_core.evidence import cell_id as _cell_id

CREDIT_LIFE_VERSION = "cl-2026.09"
SINGLE = 1
JOINT = 2

# (age_lo, age_hi, term_lo, term_hi, cover_type) -> rate per R1000 of cover. +-inf, not None:
# this ladder repeats once per cover type (see credit_core.rate_card._bands).
_INF = float("inf")
_RATES = [
    (-_INF, 30.0, -_INF, 24.0, SINGLE, 3.20), (-_INF, 30.0, 24.0, 60.0, SINGLE, 3.80), (-_INF, 30.0, 60.0, _INF, SINGLE, 4.50),
    (30.0, 55.0, -_INF, 24.0, SINGLE, 4.00), (30.0, 55.0, 24.0, 60.0, SINGLE, 4.70), (30.0, 55.0, 60.0, _INF, SINGLE, 5.60),
    (55.0, _INF, -_INF, 24.0, SINGLE, 6.50), (55.0, _INF, 24.0, 60.0, SINGLE, 7.80), (55.0, _INF, 60.0, _INF, SINGLE, 9.20),
    (-_INF, 30.0, -_INF, 24.0, JOINT, 5.40), (-_INF, 30.0, 24.0, 60.0, JOINT, 6.40), (-_INF, 30.0, 60.0, _INF, JOINT, 7.60),
    (30.0, 55.0, -_INF, 24.0, JOINT, 6.70), (30.0, 55.0, 24.0, 60.0, JOINT, 7.90), (30.0, 55.0, 60.0, _INF, JOINT, 9.40),
    (55.0, _INF, -_INF, 24.0, JOINT, 10.90), (55.0, _INF, 24.0, 60.0, JOINT, 13.10), (55.0, _INF, 60.0, _INF, JOINT, 15.40),
]


def build_credit_life_table() -> DecisionTableConfig:
    rows = [
        {"age_lo": age_lo, "age_hi": age_hi, "term_lo": term_lo, "term_hi": term_hi, "cover_type": cover, "rate": rate,
         "cell_id": _cell_id("credit_life", CREDIT_LIFE_VERSION, i)}
        for i, (age_lo, age_hi, term_lo, term_hi, cover, rate) in enumerate(_RATES)
    ]
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
