"""Stage 9 -- structuring and pricing (spec 05 §5.12), reduced to **a single lookup**
(SCOPE.md: "take structuring and pricing only as far as a single product 50 lookup").
No circular amount<->rate<->security-type solve, no candidate search: the offered
amount is capped once against a small per-grade appetite ceiling, the security type
is taken as given (no collateral cover-ratio derivation, §5.11's own scope), and the
rate is one table read.

Reuses project 00's `core.instalment` (forward calculation, unmodified),
`core.fees` (the business fee schedule -- the same flat/capped shape as the
regulated one, spec 05 §5.2's fee-cap switch is not built here) and
`core.rounding`, exactly as `credit_core`'s own demo pipeline composes them.
"""
from __future__ import annotations

from decider import dag, flow, missing_as, param, step
from decider.steps.tables import DecisionTableConfig

from credit_core import fees, instalment, rounding
from credit_core.evidence import cell_id as _cell_id

from business_nested import vocab

BUSINESS_RATE_CARD_VERSION = "biz-rate-2026.09"
DEFAULT_TERM_MONTHS = 36

# grade -> maximum facility (rand). One dimension only (SCOPE's "single lookup") --
# spec 05 §5.11's real grid is grade x sector-appetite-class x security-type (240 cells).
_APPETITE_MAX_BY_GRADE = {
    1: 10_000_000.0, 2: 9_000_000.0, 3: 8_000_000.0, 4: 7_000_000.0, 5: 6_000_000.0, 6: 5_000_000.0,
    7: 3_500_000.0, 8: 2_500_000.0, 9: 1_500_000.0, 10: 750_000.0, 11: 250_000.0, 12: 0.0,
}


def build_appetite_table() -> DecisionTableConfig:
    rows = [{"grade": g, "max_amount": amt, "cell_id": _cell_id("business_appetite", BUSINESS_RATE_CARD_VERSION, g)}
            for g, amt in _APPETITE_MAX_BY_GRADE.items()]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "business_appetite_max",
        "columns": {"grade": "Int64", "max_amount": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
        "outputs": ["max_amount", "cell_id"],
        "default": [0.0, None],
    }).relabel(writes={"cell_id": "appetite_cell_id"})


def build_rate_card() -> DecisionTableConfig:
    """grade x security_type_code -> nominal_annual_rate, one product (50 -- SCOPE.md).
    12 grades x 4 security types = 48 cells; a single lookup, not the spec's
    105 600-cell grid (amount band x term x grade x security type)."""
    rows = []
    for grade in range(1, 13):
        for security_type in (1, 2, 3, 4):
            base = 0.08 + (grade - 1) * 0.018 + (security_type - 1) * 0.01
            rows.append({
                "grade": grade, "security_type": security_type, "rate": round(base, 4),
                "cell_id": _cell_id("business_rate_card", BUSINESS_RATE_CARD_VERSION, grade, security_type),
            })
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "business_rate_card",
        "columns": {"grade": "Int64", "security_type": "Int64", "rate": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
            {"type": "eq", "variable": "security_type_code", "value_column": "security_type"},
        ]},
        "outputs": ["rate", "cell_id"],
        "default": [0.30, None],
    }).relabel(writes={"rate": "nominal_annual_rate", "cell_id": "rate_card_cell_id"})


def term_months(requested_term_months: int = missing_as(DEFAULT_TERM_MONTHS)) -> int:
    return requested_term_months if requested_term_months and requested_term_months > 0 else DEFAULT_TERM_MONTHS


def offered_amount(requested_amount: float, max_amount: float) -> float:
    return round(min(requested_amount, max_amount) / 100.0) * 100.0  # core.rounding.round_advance, inline shape


def binding_constraint_code(requested_amount: float, offered_amount: float, max_amount: float) -> str:
    if offered_amount >= requested_amount:
        return "requested_amount"
    return "appetite_grade_cap" if offered_amount >= max_amount - 1e-6 else "product_minimum"


def _instalment_unit():
    """A same-name waterfall (`instalment` computed, then rounded to the cent under the
    same name) has to be a `flow`, not a `dag` -- exactly 00's own `pipeline.py` docstring
    for `_instalment_unit()`, reused here for the identical reason."""
    return flow(
        instalment.instalment_before_fees_step,
        instalment.instalment_step,
        rounding.round_instalment_step.relabel(reads={"amount": "instalment"}),
        instalment.total_cost_of_credit_step,
        instalment.effective_annual_rate_step,
        name="instalment",
    ).emit("instalment_before_fees")


def build_pricing_unit():
    return dag(
        build_appetite_table(),
        step(offered_amount),
        step(binding_constraint_code),
        step(term_months),
        build_rate_card(),
        fees.initiation_fee_step, fees.monthly_service_fee_step,
        _instalment_unit(),
        name="pricing",
    ).emit(
        "max_amount", "appetite_cell_id", "offered_amount", "binding_constraint_code", "term_months",
        "nominal_annual_rate", "rate_card_cell_id", "initiation_fee", "monthly_service_fee",
        "instalment_before_fees", "instalment", "total_cost_of_credit", "effective_annual_rate",
    )
