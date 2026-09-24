"""preassessment.py: Stage 5 (spec 04 §5.5) -- pre-assessment reusing project 03's published
solve and pricing, and the §5.5 "Emits" contract fields."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from decider import Engine

from campaign_trees import preassessment
from loan_granting.pricing import CreditLifeIndex, PriceEvaluator, RateCardIndex
from loan_granting.solve import solve_term


def _rate_card_index() -> RateCardIndex:
    from credit_core.rate_card import generate_flex_loan_card
    from decider.steps.tables import DecisionTableConfig
    doc = generate_flex_loan_card("rc-test")
    table = DecisionTableConfig.load(doc)
    return RateCardIndex.from_configurable_step(table)


def test_estimate_max_affordable_instalment_is_a_fraction_of_discretionary_income():
    exe = Engine().bind(preassessment.estimate_max_affordable_instalment, mode="interpreted")
    out = exe.score({"estimated_discretionary_income": 3000.0}, params={
        "estimate_max_affordable_instalment": {"serviceability_ratio": 0.30}})
    assert out["estimate_max_affordable_instalment"] == 900.0


def test_preassessment_reuses_03s_solve_and_pricing_directly():
    """Not a re-implementation: this project's step calls `loan_granting.solve.solve_term`
    and `loan_granting.pricing.PriceEvaluator` -- the exact objects 03 publishes for reuse
    (03 NOTES.md "What I publish") -- so this test proves the *same* call both modules make
    produces the amount this project reports."""
    rate_card_index = _rate_card_index()
    credit_life_index = CreditLifeIndex()
    evaluator = PriceEvaluator(rate_card_index, credit_life_index, statutory_ceiling=0.28,
                                credit_life_substitution_declared=False)

    direct = solve_term(evaluator, 36, risk_grade=5, applicant_age_years=41.0, is_joint=False,
                         amount_cap=325_000.0, requested_amount=None, max_affordable_instalment=935.4)

    step = preassessment.build_preassessment_step(rate_card_index, credit_life_index, 0.28)
    df = pl.DataFrame([{
        "risk_grade": 5, "applicant_age_years": 41.0, "max_affordable_instalment": 935.4,
        "appetite_max_amount": 325_000.0, "cycle_date": date(2026, 9, 24), "is_daily_delta": False,
    }])
    out = step(df)
    assert out["pre_assessed_amount"][0] == direct.amount
    assert out["binding_constraint_code"][0] == direct.binding_constraint_code


def test_daily_delta_gets_the_shorter_validity_window():
    rate_card_index = _rate_card_index()
    credit_life_index = CreditLifeIndex()
    step = preassessment.build_preassessment_step(rate_card_index, credit_life_index, 0.28)
    df = pl.DataFrame([
        {"risk_grade": 5, "applicant_age_years": 41.0, "max_affordable_instalment": 935.4,
         "appetite_max_amount": 325_000.0, "cycle_date": date(2026, 9, 24), "is_daily_delta": False},
        {"risk_grade": 5, "applicant_age_years": 41.0, "max_affordable_instalment": 935.4,
         "appetite_max_amount": 325_000.0, "cycle_date": date(2026, 9, 24), "is_daily_delta": True},
    ])
    out = step(df)
    monthly_window = out["pre_assessment_valid_to"][0] - out["pre_assessment_valid_from"][0]
    daily_window = out["pre_assessment_valid_to"][1] - out["pre_assessment_valid_from"][1]
    assert monthly_window == timedelta(days=35)
    assert daily_window == timedelta(days=10)


def test_is_preassessment_expired():
    assert preassessment.is_preassessment_expired(date(2026, 11, 1), date(2026, 10, 29)) is True
    assert preassessment.is_preassessment_expired(date(2026, 10, 1), date(2026, 10, 29)) is False
