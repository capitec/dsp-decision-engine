"""The amount the bank would grant this client today: the largest affordable Flex Loan, from the granting example's search."""
from __future__ import annotations

import bisect
from datetime import date, timedelta

import numpy as np
from decider import DecisionTableConfig, ParamTable, allow_fallback, missing_as, param, param_table, step

from credit_core.credit_life import JOINT, SINGLE, build_credit_life_table
from loan_granting.pricing import PriceEvaluator
from loan_granting.solve import solve_term

TERM_MONTHS = 36  # a campaign advertises an amount, not a menu of terms

CARD_COLUMNS = {"grade": int, "amt_lo": float, "term_lo": float, "rate": float}
CREDIT_LIFE_COLUMNS = {"cover_type": int, "age_lo": float, "age_hi": float, "term_lo": float, "term_hi": float, "rate": float}


def max_affordable_instalment(
    estimated_discretionary_income: float = missing_as(0.0), serviceability_ratio: float = param(0.30, ge=0.05, le=0.60),
) -> float:
    # ponytail: a share of the mart's discretionary-income estimate; the affordability example's assessment is the real one
    return round(max(0.0, estimated_discretionary_income) * serviceability_ratio, 2)


class _Card:
    """The interface `PriceEvaluator` uses of `loan_granting.pricing.RateCardIndex`, over a table sorted by grade, band, term."""

    version = None

    def __init__(self, card: ParamTable):
        self.card = card
        self._grades: dict[int, tuple] = {}

    def _grade(self, grade: int) -> tuple:
        # One search per grade and a list of band starts: a solve looks up a dozen amounts for the same grade.
        if grade not in self._grades:
            g0, g1 = (int(np.searchsorted(self.card.grade, grade, side)) for side in ("left", "right"))
            amounts = self.card.amt_lo[g0:g1]
            new_band = np.concatenate(([True], amounts[1:] != amounts[:-1])) if g1 > g0 else np.zeros(0, bool)
            first = np.flatnonzero(new_band)
            self._grades[grade] = (g0, g1, first, amounts[first].tolist(), (first + g0).tolist() + [g1])
        return self._grades[grade]

    def lookup(self, amount: float, term_months: float, grade: int) -> tuple[float, str] | None:
        _, _, _, starts, bounds = self._grade(grade)
        i = bisect.bisect_right(starts, amount) - 1
        if i < 0:
            return None
        j = bisect.bisect_right(self.card.term_lo[bounds[i]:bounds[i + 1]].tolist(), term_months) - 1
        return (float(self.card.rate[bounds[i] + j]), "") if j >= 0 else None

    def amount_bands(self, term_months: float, grade: int) -> list[tuple[float, float | None]]:
        g0, g1, first, starts, _ = self._grade(grade)
        if not starts:
            return []
        within = np.add.reduceat(self.card.term_lo[g0:g1] <= term_months, first, dtype=np.int64)  # a prefix of each band
        rates = self.card.rate[g0 + first + np.maximum(within, 1) - 1].tolist()
        return [(start, rate if n else None) for start, rate, n in zip(starts, rates, within.tolist())]


class _CreditLife:
    """The interface `PriceEvaluator` uses of `loan_granting.pricing.CreditLifeIndex`, over a table of cover rates."""

    def __init__(self, table: ParamTable):
        self.rows = list(zip(*(getattr(table, name).tolist() for name in CREDIT_LIFE_COLUMNS)))

    def lookup(self, applicant_age_years: float, term_months: float, is_joint: bool) -> tuple[float, str | None]:
        cover = JOINT if is_joint else SINGLE
        for cover_type, age_lo, age_hi, term_lo, term_hi, rate in self.rows:
            if cover_type == cover and age_lo <= applicant_age_years < age_hi and term_lo <= term_months < term_hi:
                return rate, ""
        return 0.0, None


def card_rows(rate_card: DecisionTableConfig) -> list[dict]:
    return rate_card.rows.df.sort("grade", "amt_lo", "term_lo").select(list(CARD_COLUMNS)).to_dicts()


def credit_life_rows() -> list[dict]:
    return build_credit_life_table().rows.df.select(list(CREDIT_LIFE_COLUMNS)).to_dicts()


def pre_assessment_step(rate_card: DecisionTableConfig):
    """The step writing `pre_assessed_amount`, its term, what bound it, and the dates the pre-assessment is valid.

    The rate card and the credit life rates are table params whose defaults are the rows of the documents, so a
    params document can replace either one and a diff of two builds sees the rows.
    """
    card_default, credit_life_default = card_rows(rate_card), credit_life_rows()

    @allow_fallback
    @step(outputs=("pre_assessed_amount", "pre_assessed_term", "binding_constraint_code", "pre_assessment_evaluations",
                   "pre_assessment_valid_from", "pre_assessment_valid_to"))
    def pre_assessment(
        risk_grade: float, applicant_age_years: float, max_affordable_instalment: float, appetite_max_amount: float,
        cycle_date: date, is_daily_delta: bool = missing_as(False), statutory_ceiling: float = param(0.28, gt=0.0),
        monthly_validity_days: int = param(35, ge=1), daily_validity_days: int = param(10, ge=1),
        flex_loan_rate_card: ParamTable = param_table(CARD_COLUMNS, default=card_default),
        credit_life_rates: ParamTable = param_table(CREDIT_LIFE_COLUMNS, default=credit_life_default),
    ) -> tuple[float, int, str, int, date, date]:
        evaluator = PriceEvaluator(_Card(flex_loan_rate_card), _CreditLife(credit_life_rates), statutory_ceiling,
                                   credit_life_substitution_declared=False)
        # No requested amount: an unsolicited offer is capped by the bank's appetite alone.
        result = solve_term(evaluator, TERM_MONTHS, int(risk_grade), applicant_age_years, False, appetite_max_amount, None,
                            max_affordable_instalment)
        days = daily_validity_days if is_daily_delta else monthly_validity_days
        return (result.amount or 0.0, TERM_MONTHS, result.binding_constraint_code, result.evaluation_count,
                cycle_date, cycle_date + timedelta(days=days))

    return pre_assessment
