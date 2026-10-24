"""Pricing: the offer, from the grade. The amount is capped by appetite, the rate is read from a card, and the
instalment is the annuity payment plus fees."""
from __future__ import annotations

from decikit import DecisionTableConfig, dag, flow, missing_as, param, step

from credit_core import fees, instalment, rounding
from credit_core.evidence import cell_id

RATE_CARD_VERSION = "biz-rate-2026.09"

# binding_constraint_code: why the offer is not larger.
REQUESTED_AMOUNT = 1
APPETITE_CAP = 2

# grade -> the largest facility the bank will write at that grade, in rand.
_MAX_AMOUNT = {1: 10_000_000.0, 2: 9_000_000.0, 3: 8_000_000.0, 4: 7_000_000.0, 5: 6_000_000.0, 6: 5_000_000.0,
               7: 3_500_000.0, 8: 2_500_000.0, 9: 1_500_000.0, 10: 750_000.0, 11: 250_000.0, 12: 0.0}


def build_appetite_table() -> DecisionTableConfig:
    """Writes `max_amount` and `appetite_cell_id` for `risk_grade`; an unlisted grade gets nothing."""
    rows = [{"grade": g, "max_amount": amount, "cell_id": cell_id("business_appetite", RATE_CARD_VERSION, g)}
            for g, amount in _MAX_AMOUNT.items()]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "business_appetite",
        "columns": {"grade": "Int64", "max_amount": "Float64", "cell_id": "String"}, "rows": rows,
        "expression": {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
        "outputs": ["max_amount", "cell_id"], "default": [0.0, None],
    }).relabel(writes={"cell_id": "appetite_cell_id"})


def build_rate_card() -> DecisionTableConfig:
    """Writes `nominal_annual_rate` and `rate_card_cell_id` for `risk_grade` and `security_type_code` (1 to 4).

    The rate rises 1.8 points a grade and one point a security type; any other combination pays 30%.
    """
    rows = [{"grade": g, "security": s, "rate": round(0.08 + (g - 1) * 0.018 + (s - 1) * 0.01, 4),
             "cell_id": cell_id("business_rate_card", RATE_CARD_VERSION, g, s)}
            for g in range(1, 13) for s in (1, 2, 3, 4)]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "business_rate_card",
        "columns": {"grade": "Int64", "security": "Int64", "rate": "Float64", "cell_id": "String"}, "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
            {"type": "eq", "variable": "security_type_code", "value_column": "security"}]},
        "outputs": ["rate", "cell_id"], "default": [0.30, None],
    }).relabel(writes={"rate": "nominal_annual_rate", "cell_id": "rate_card_cell_id"})


def capped_amount(requested_amount: float, max_amount: float) -> float:
    return min(requested_amount, max_amount)


def binding_constraint_code(requested_amount: float, max_amount: float) -> int:
    return APPETITE_CAP if max_amount < requested_amount else REQUESTED_AMOUNT


def term_months(requested_term_months: int = missing_as(0), default_term_months: int = param(36, ge=1)) -> int:
    return requested_term_months if requested_term_months > 0 else default_term_months


def pricing_unit():
    """Appetite, rate, fees and instalment for the `risk_grade`; reads `requested_amount`, `security_type_code`
    and, optionally, `requested_term_months`."""
    # `instalment` is written twice, with fees added and then rounded to the cent, so it is a flow.
    instalment_unit = flow(
        instalment.instalment_before_fees_step, instalment.instalment_step,
        rounding.round_instalment_step.relabel(reads={"amount": "instalment"}),
        instalment.total_cost_of_credit_step, instalment.effective_annual_rate_step,
        name="instalment").emit("instalment")
    return dag(
        build_appetite_table(), step(capped_amount), step(binding_constraint_code), step(term_months),
        rounding.round_advance_step.relabel(reads={"amount": "capped_amount"}),
        build_rate_card(), fees.initiation_fee_step, fees.monthly_service_fee_step, instalment_unit,
        name="pricing",
    ).emit("max_amount", "appetite_cell_id", "offered_amount", "binding_constraint_code", "term_months",
           "nominal_annual_rate", "rate_card_cell_id", "initiation_fee", "monthly_service_fee", "instalment",
           "total_cost_of_credit", "effective_annual_rate")
