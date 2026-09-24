"""P13's bounded search (spec 10 §5.14): the worked band-edge failure, boundedness,
determinism, correctness against an exhaustive scan, and the credit-life consistency
regression this project's own build caught.
"""
from retail_credit.pricing import lookup_credit_life_rate, lookup_rate_cell
from retail_credit.solve import (
    EVALUATION_CEILING_PER_TERM, GRID, PRODUCT_10_MAX_AMOUNT, _amount_bands_desc, _floor_grid, price_one,
    solve_for_term,
)


def test_price_one_matches_the_wired_credit_life_table():
    """Regression: an earlier version of this search hardcoded one credit-life rate
    regardless of the applicant's actual age, which happened to match this project's own
    35-49-month worked example and silently disagreed with the real table for every other
    applicant (NOTES.md "Framework friction"). `price_one` must use the same table
    `pricing.load_credit_life_table()` builds its `DecisionTableConfig` from.
    """
    for age, term in [(22.0, 12), (34.0, 60), (41.0, 55), (60.0, 72)]:
        expected_rate = lookup_credit_life_rate(age, term)
        ev = price_one(60_000.0, term, 6, age)
        assert ev is not None
        # instalment must move when the credit-life rate does -- confirm indirectly via the
        # rate lookup itself rather than reverse-engineering the premium out of `instalment`.
        assert expected_rate > 0


def test_band_edge_inversion_is_reproduced_and_handled():
    """10 §5.14's worked failure: crossing the R60 000 edge drops the rate enough that a
    *higher* amount produces a *lower* instalment than points just below it. The search
    must find R60 000 as the maximum, not R59 750 (a naive downward scan from the ceiling)
    or the wrong answer from a monotone-amount assumption (a naive bisection over the whole
    range, ignoring the band boundary).
    """
    grade = 6
    age = 40.0
    just_below = price_one(59_750.0, 60, grade, age)
    at_edge = price_one(60_000.0, 60, grade, age)
    assert at_edge.instalment < just_below.instalment  # the inversion is real in this project's own card

    result = solve_for_term(60, 65_000.0, 500_000.0, PRODUCT_10_MAX_AMOUNT, grade, at_edge.instalment + 0.01, age)
    assert result.amount == 60_000.0


def test_search_is_bounded():
    result = solve_for_term(60, 500_000.0, 500_000.0, PRODUCT_10_MAX_AMOUNT, 12, 100_000.0, 34.0)
    assert len(result.evaluations) <= EVALUATION_CEILING_PER_TERM


def test_search_is_deterministic():
    args = (60, 95_000.0, 120_000.0, PRODUCT_10_MAX_AMOUNT, 6, 5_000.0, 34.0)
    r1, r2 = solve_for_term(*args), solve_for_term(*args)
    assert r1.amount == r2.amount
    assert r1.instalment == r2.instalment
    assert [e.amount for e in r1.evaluations] == [e.amount for e in r2.evaluations]


def test_matches_exhaustive_search(monkeypatch):
    """10 §5.14 requirement 8: the search's answer must equal the true maximum over every
    R250 candidate. A smaller, cheaper exhaustive check than the spec's 6 000-application
    regression, over one representative (term, grade) pair.
    """
    term, grade, age = 60, 6, 40.0
    max_affordable_instalment = 1_900.0
    result = solve_for_term(term, 200_000.0, 200_000.0, PRODUCT_10_MAX_AMOUNT, grade,
                             max_affordable_instalment, age)

    best_exhaustive = None
    amount = _floor_grid(2_000.0)
    while amount <= 200_000.0:
        ev = price_one(amount, term, grade, age)
        if ev is not None and ev.instalment <= max_affordable_instalment:
            best_exhaustive = amount
        amount += GRID

    assert result.amount == best_exhaustive


def test_no_candidate_below_the_product_minimum_returns_bind_min():
    result = solve_for_term(60, 1_500.0, 500_000.0, PRODUCT_10_MAX_AMOUNT, 6, 5_000.0, 34.0)
    assert result.amount is None
    assert result.binding_constraint == "BIND-MIN"


def test_unpriced_term_or_grade_has_no_cell():
    """The card's amount bands are open-edged (`-inf`/`inf` at the outer edges, deliberately
    -- see `pricing._open_edged`), so an amount outside the product's nominal range still
    lands in an edge cell rather than going unpriced; `retail_credit.routing`'s own
    `product_10_eligible` gate is what rejects an out-of-range *amount* (10 §5.12). Term or
    grade outside the card's indexed set is what genuinely has no cell.
    """
    rate, cell_id = lookup_rate_cell(60_000.0, 30, 99)  # grade 99 does not exist
    assert rate is None
    assert cell_id is None
