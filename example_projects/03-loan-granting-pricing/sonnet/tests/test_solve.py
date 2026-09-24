"""The bounded solve (spec 03 §5.8): the named non-monotonicity regression,
the evaluation ceiling, determinism, and agreement with exhaustive search."""
from __future__ import annotations

from credit_core.rate_card import generate_flex_loan_card
from loan_granting.pricing import CreditLifeIndex, PriceEvaluator, RateCardIndex
from loan_granting.solve import BIND_EXH, exhaustive_search, solve_term

INF = float("inf")


def _inversion_evaluator():
    """A hand-built card with a deliberate band-edge inversion (spec 03 §5.8's worked
    failure): the amount band below R50 000 prices *worse* than the one starting at it.
    Project 00's own generated card has no such inversion (its rate formula is strictly
    increasing in amount -- see NOTES.md "Gaps in what I consumed"), so this fixture is
    the only way to exercise the property spec 03 §10 item 6 requires as a named test."""
    rows = [
        {"amt_lo": -INF, "amt_hi": 49_000.0, "term_lo": -INF, "term_hi": INF, "grade": 9, "rate": 19.75,
         "cell_id": "c1", "rate_card_version": "v1"},
        {"amt_lo": 49_000.0, "amt_hi": 50_000.0, "term_lo": -INF, "term_hi": INF, "grade": 9, "rate": 19.75,
         "cell_id": "c2", "rate_card_version": "v1"},
        {"amt_lo": 50_000.0, "amt_hi": INF, "term_lo": -INF, "term_hi": INF, "grade": 9, "rate": 18.25,
         "cell_id": "c3", "rate_card_version": "v1"},
    ]
    idx = RateCardIndex(rows)
    return PriceEvaluator(idx, CreditLifeIndex(), statutory_ceiling=0.32, credit_life_substitution_declared=True)


def test_band_edge_non_monotonicity_finds_the_true_maximum():
    """A plain bisection over the whole domain (assuming monotone affordability) would
    stop inside the lower, worse-priced band and under-lend -- spec 03 §5.8's own
    stated failure mode. The banded search must not do that."""
    ev = _inversion_evaluator()
    result = solve_term(ev, term_months=60, risk_grade=9, applicant_age_years=38.0, is_joint=False,
                         amount_cap=500_000.0, requested_amount=None, max_affordable_instalment=1_560.00)
    exhaustive = exhaustive_search(ev, term_months=60, risk_grade=9, applicant_age_years=38.0, is_joint=False,
                                    amount_cap=500_000.0, requested_amount=None, max_affordable_instalment=1_560.00)
    assert result.amount == exhaustive
    # The feasible set here is {2000..48500} u {50000..53000}: the true maximum sits
    # *above* the band edge, reachable only by evaluating into the higher band.
    assert result.amount == 53_000.0
    assert result.evaluation_count <= 24


def test_solve_is_deterministic():
    ev = _inversion_evaluator()
    results = [
        solve_term(ev, 60, 9, 38.0, False, 500_000.0, None, 1_560.00).amount
        for _ in range(5)
    ]
    assert len(set(results)) == 1


def test_evaluation_ceiling_is_never_exceeded_and_refers_when_reached():
    ev = _inversion_evaluator()
    result = solve_term(ev, term_months=60, risk_grade=9, applicant_age_years=38.0, is_joint=False,
                         amount_cap=500_000.0, requested_amount=None, max_affordable_instalment=1_560.00,
                         evaluation_ceiling=2)
    assert result.evaluation_count <= 2
    assert result.amount is None
    assert result.binding_constraint_code == BIND_EXH


def test_domain_top_feasible_is_the_fast_path_one_evaluation():
    ev = _inversion_evaluator()
    result = solve_term(ev, term_months=60, risk_grade=9, applicant_age_years=38.0, is_joint=False,
                         amount_cap=40_000.0, requested_amount=None, max_affordable_instalment=100_000.0)
    assert result.amount == 40_000.0
    assert result.evaluation_count == 1


def test_agreement_with_exhaustive_search_over_the_real_flex_loan_card():
    """Spec 03 §10 item 3: the solve's answer must equal the answer from evaluating
    every R100 candidate, over a sample of realistic inputs and the real generated card."""
    doc = generate_flex_loan_card("rc-test")
    idx = RateCardIndex(doc["rows"])
    ev = PriceEvaluator(idx, CreditLifeIndex(), statutory_ceiling=0.32, credit_life_substitution_declared=False)
    cases = [
        (60, 7, 38.0, 95_000.0, None, 2_800.0), (12, 3, 45.0, 45_000.0, 120_000.0, 4_000.0),
        (84, 10, 29.0, 30_000.0, None, 900.0), (24, 1, 55.0, 500_000.0, None, 15_000.0),
        (6, 12, 33.0, 2_500.0, None, 100.0),
    ]
    for term, grade, age, amount_cap, requested, max_inst in cases:
        got = solve_term(ev, term, grade, age, False, amount_cap, requested, max_inst)
        want = exhaustive_search(ev, term, grade, age, False, amount_cap, requested, max_inst)
        assert got.amount == want, (term, grade, age, amount_cap, requested, max_inst, got.amount, want)
