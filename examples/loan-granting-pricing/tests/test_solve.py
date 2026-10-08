"""The solve against brute force: every R100 candidate priced, on cards where a smaller amount can cost more."""
import random

from credit_core.rate_card import generate_flex_loan_card

from loan_granting.pricing import INF, CreditLifeIndex, PriceEvaluator, RateCardIndex
from loan_granting.solve import (
    BIND_AFF, BIND_CAP, BIND_CEIL, BIND_EXH, BIND_MIN, BIND_REQ, DEFAULT_EVALUATION_CEILING, exhaustive_search,
    solve_term,
)


def card(*bands, grade=9, version="v1"):
    """A card for one grade and every term, from `(lower edge in rand, rate in percent)` bands."""
    return RateCardIndex([{"amt_lo": -INF if i == 0 else edge, "term_lo": -INF, "grade": grade, "rate": rate,
                           "cell_id": f"cell-{i}", "rate_card_version": version}
                          for i, (edge, rate) in enumerate(bands)])


def evaluator(index, ceiling=0.32, own_credit_life=True):
    return PriceEvaluator(index, CreditLifeIndex(), ceiling, own_credit_life)


def solve(ev, affordable, term=60, cap=500_000.0, requested=None, ceiling=DEFAULT_EVALUATION_CEILING):
    return solve_term(ev, term, 9, 38.0, False, cap, requested, affordable, ceiling)


def brute_force(ev, affordable, term=60, cap=500_000.0, requested=None):
    return exhaustive_search(ev, term, 9, 38.0, False, cap, requested, affordable)


# Treasury rewards crossing R50 000 with 150 basis points, so affordability is not monotone in the amount.
REWARD_AT_50K = card((0, 19.75), (50_000.0, 18.25))


def test_the_maximum_can_sit_above_a_cheaper_band_edge():
    ev = evaluator(REWARD_AT_50K)
    result = solve(ev, affordable=1_500.0)
    assert result.amount == 50_900.0
    below_the_edge = max(a * 100.0 for a in range(20, 500) if ev.evaluate(a * 100.0, 60, 9, 38.0, False).instalment <= 1_500.0)
    assert below_the_edge == 49_300.0  # where a bisection that assumes monotone affordability stops
    assert result.binding_constraint_code == BIND_AFF
    assert result.pricing.instalment <= 1_500.0 < ev.evaluate(51_000.0, 60, 9, 38.0, False).instalment


def bisection_assuming_monotone(ev, affordable, term, cap, requested):
    """The tempting search: halve the whole range as if a smaller amount always cost less."""
    lo, hi, best = 20, int(min(cap, requested or cap) // 100), None
    while lo <= hi:
        mid = (lo + hi) // 2
        if ev.evaluate(mid * 100.0, term, 9, 38.0, False).instalment <= affordable:
            best, lo = mid * 100.0, mid + 1
        else:
            hi = mid - 1
    return best


def test_every_solve_agrees_with_pricing_every_candidate_on_cards_with_inversions():
    rng, plain_bisection_wrong = random.Random(7), 0
    for _ in range(300):
        edges = sorted(rng.sample(range(3_000, 120_000, 500), rng.randint(2, 6)))
        ev = evaluator(card((0, rng.uniform(12, 28)), *[(float(e), rng.uniform(12, 28)) for e in edges]),
                       own_credit_life=False)
        term, cap = rng.choice([12, 24, 36, 48, 60, 72, 84]), rng.randrange(3_000, 120_000, 100) * 1.0
        requested = rng.choice([None, rng.randrange(2_000, 120_000, 100) * 1.0])
        affordable = ev.evaluate(rng.randrange(2_000, int(cap) + 100, 100) * 1.0, term, 9, 38.0, False).instalment
        got = solve(ev, affordable, term, cap, requested)
        assert got.amount == brute_force(ev, affordable, term, cap, requested), (term, cap, requested, affordable)
        assert got.evaluation_count <= DEFAULT_EVALUATION_CEILING
        plain_bisection_wrong += bisection_assuming_monotone(ev, affordable, term, cap, requested) != got.amount
    assert plain_bisection_wrong >= 5  # the cases are hard enough that a monotone assumption fails on them


def test_every_solve_agrees_with_pricing_every_candidate_on_the_generated_card():
    index = RateCardIndex(generate_flex_loan_card("rc-test")["rows"])
    ev = PriceEvaluator(index, CreditLifeIndex(), 0.32, False)
    rng = random.Random(11)
    for _ in range(25):
        grade, term = rng.randint(1, 12), rng.randint(6, 84)
        cap, affordable = rng.randrange(5_000, 150_000, 100) * 1.0, round(rng.uniform(150.0, 6_000.0), 2)
        got = solve_term(ev, term, grade, 41.0, False, cap, None, affordable)
        assert got.amount == exhaustive_search(ev, term, grade, 41.0, False, cap, None, affordable), (grade, term)
        assert got.evaluation_count <= DEFAULT_EVALUATION_CEILING


def test_a_card_whose_rate_falls_at_every_edge_is_solved_exactly_or_reported_exhausted():
    falling = card(*[(2_000.0 + i * 5_000.0, 24.0 - i * 0.1) for i in range(96)])
    ev = evaluator(falling)
    for affordable in (400.0, 900.0, 1_700.0):
        got = solve(ev, affordable)
        assert got.evaluation_count <= DEFAULT_EVALUATION_CEILING
        if got.amount is None:
            assert got.binding_constraint_code == BIND_EXH
        else:
            assert got.amount == brute_force(ev, affordable)


def test_reaching_the_evaluation_ceiling_yields_no_amount_rather_than_a_guess():
    result = solve(evaluator(REWARD_AT_50K), affordable=1_500.0, ceiling=3)
    assert (result.amount, result.binding_constraint_code, result.evaluation_count) == (None, BIND_EXH, 3)


def test_an_affordable_request_is_one_evaluation_and_bound_by_the_request():
    result = solve(evaluator(REWARD_AT_50K), affordable=10_000.0, requested=30_000.0)
    assert (result.amount, result.binding_constraint_code, result.evaluation_count) == (30_000.0, BIND_REQ, 1)


def test_the_cap_binds_when_it_is_below_the_request_and_affordable():
    result = solve(evaluator(REWARD_AT_50K), affordable=10_000.0, cap=40_000.0, requested=90_000.0)
    assert (result.amount, result.binding_constraint_code) == (40_000.0, BIND_CAP)


def test_nothing_affordable_gives_no_amount():
    result = solve(evaluator(REWARD_AT_50K), affordable=10.0)
    assert (result.amount, result.binding_constraint_code) == (None, BIND_AFF)


def test_a_cap_or_request_under_the_product_minimum_leaves_no_domain():
    ev = evaluator(REWARD_AT_50K)
    assert solve(ev, 5_000.0, cap=1_900.0).binding_constraint_code == BIND_CAP
    assert solve(ev, 5_000.0, requested=1_900.0).binding_constraint_code == BIND_MIN
    assert solve(ev, 5_000.0, cap=1_900.0).evaluation_count == 0


def test_cells_above_the_statutory_ceiling_are_never_offered():
    ev = evaluator(card((0, 19.0), (60_000.0, 31.0)), ceiling=0.28)
    result = solve(ev, affordable=10_000.0)
    assert result.amount == 59_900.0
    assert solve(evaluator(card((0, 31.0)), ceiling=0.28), 10_000.0).binding_constraint_code == BIND_CEIL


def test_the_same_inputs_give_the_same_answer_every_time():
    ev = evaluator(REWARD_AT_50K)
    answers = {(r.amount, r.evaluation_count) for r in (solve(ev, 1_500.0) for _ in range(5))}
    assert len(answers) == 1


def test_the_winning_pricing_is_what_a_fresh_evaluation_of_that_amount_gives():
    ev = evaluator(REWARD_AT_50K)
    result = solve(ev, affordable=1_500.0)
    assert result.pricing == ev.evaluate(result.amount, 60, 9, 38.0, False)
