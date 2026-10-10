"""The amount the bank would grant this client today: the largest affordable Flex Loan, from the granting example's search."""
from __future__ import annotations

from datetime import date, timedelta

from decikit import DecisionTableConfig, ParamTable, allow_fallback, missing_as, param, param_table, step

from credit_core.credit_life import CREDIT_LIFE_VERSION
from loan_granting.pricing import PriceEvaluator
from loan_granting.rate_tables import (
    CREDIT_LIFE_COLUMNS, RATE_CARD_COLUMNS, CreditLifeTable, RateCardTable, credit_life_rows, rate_card_rows,
)
from loan_granting.solve import solve_term

TERM_MONTHS = 36  # a campaign advertises an amount, not a menu of terms


def max_affordable_instalment(
    estimated_discretionary_income: float = missing_as(0.0), serviceability_ratio: float = param(0.30, ge=0.05, le=0.60),
) -> float:
    # ponytail: a share of the mart's discretionary-income estimate; the affordability example's assessment is the real one
    return round(max(0.0, estimated_discretionary_income) * serviceability_ratio, 2)


def pre_assessment_step(rate_card: DecisionTableConfig):
    """The step writing `pre_assessed_amount`, its term, what bound it, and the dates the pre-assessment is valid.

    The rate card and the credit life rates are table params whose defaults are the rows of the documents, so a
    params document can replace either one and a diff of two builds sees the rows.
    """
    card_default, card_version_default = rate_card_rows(rate_card)

    @allow_fallback
    @step(outputs=("pre_assessed_amount", "pre_assessed_term", "binding_constraint_code", "pre_assessment_evaluations",
                   "pre_assessment_valid_from", "pre_assessment_valid_to"))
    def pre_assessment(
        risk_grade: float, applicant_age_years: float, max_affordable_instalment: float, appetite_max_amount: float,
        cycle_date: date, is_daily_delta: bool = missing_as(False), statutory_ceiling: float = param(0.28, gt=0.0),
        monthly_validity_days: int = param(35, ge=1), daily_validity_days: int = param(10, ge=1),
        flex_loan_rate_card: ParamTable = param_table(RATE_CARD_COLUMNS, default=card_default),
        flex_loan_rate_card_version: str = param(card_version_default),
        credit_life_rates: ParamTable = param_table(CREDIT_LIFE_COLUMNS, default=credit_life_rows()),
        credit_life_version: str = param(CREDIT_LIFE_VERSION),
    ) -> tuple[float, int, str, int, date, date]:
        evaluator = PriceEvaluator(RateCardTable(flex_loan_rate_card, flex_loan_rate_card_version),
                                   CreditLifeTable(credit_life_rates, credit_life_version), statutory_ceiling,
                                   credit_life_substitution_declared=False)
        # No requested amount: an unsolicited offer is capped by the bank's appetite alone.
        result = solve_term(evaluator, TERM_MONTHS, int(risk_grade), applicant_age_years, False, appetite_max_amount, None,
                            max_affordable_instalment)
        days = daily_validity_days if is_daily_delta else monthly_validity_days
        return (result.amount or 0.0, TERM_MONTHS, result.binding_constraint_code, result.evaluation_count,
                cycle_date, cycle_date + timedelta(days=days))

    return pre_assessment
