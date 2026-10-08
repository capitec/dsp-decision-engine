"""The amount the bank would grant this client today: the largest affordable Flex Loan, from the granting example's search."""
from __future__ import annotations

from datetime import date, timedelta

from decider import allow_fallback, missing_as, param, step

from loan_granting.pricing import CreditLifeIndex, PriceEvaluator, RateCardIndex
from loan_granting.solve import solve_term

TERM_MONTHS = 36  # a campaign advertises an amount, not a menu of terms


def max_affordable_instalment(
    estimated_discretionary_income: float = missing_as(0.0), serviceability_ratio: float = param(0.30, ge=0.05, le=0.60),
) -> float:
    # ponytail: a share of the mart's discretionary-income estimate; the affordability example's assessment is the real one
    return round(max(0.0, estimated_discretionary_income) * serviceability_ratio, 2)


def pre_assessment_step(rate_card_index: RateCardIndex):
    """The step writing `pre_assessed_amount`, its term, what bound it, and the dates the pre-assessment is valid."""
    credit_life_index = CreditLifeIndex()

    @allow_fallback
    @step(outputs=("pre_assessed_amount", "pre_assessed_term", "binding_constraint_code", "pre_assessment_evaluations",
                   "pre_assessment_valid_from", "pre_assessment_valid_to"))
    def pre_assessment(
        risk_grade: float, applicant_age_years: float, max_affordable_instalment: float, appetite_max_amount: float,
        cycle_date: date, is_daily_delta: bool = missing_as(False), statutory_ceiling: float = param(0.28, gt=0.0),
        monthly_validity_days: int = param(35, ge=1), daily_validity_days: int = param(10, ge=1),
    ) -> tuple[float, int, str, int, date, date]:
        evaluator = PriceEvaluator(rate_card_index, credit_life_index, statutory_ceiling, credit_life_substitution_declared=False)
        # No requested amount: an unsolicited offer is capped by the bank's appetite alone.
        result = solve_term(evaluator, TERM_MONTHS, int(risk_grade), applicant_age_years, False, appetite_max_amount, None,
                            max_affordable_instalment)
        days = daily_validity_days if is_daily_delta else monthly_validity_days
        return (result.amount or 0.0, TERM_MONTHS, result.binding_constraint_code, result.evaluation_count,
                cycle_date, cycle_date + timedelta(days=days))

    return pre_assessment
