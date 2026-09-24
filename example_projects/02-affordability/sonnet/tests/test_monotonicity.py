"""Acceptance §10 item 3 / spec 02 §5.7.2(c): the verdict is monotone in
`proposed_instalment` -- a larger proposed instalment can never produce a
better verdict. Project 03's search has no valid stopping condition without
this. Swept across band edges, buffer boundaries and the residual floor,
not merely asserted at one point."""
from decider import Engine

from assessment import capacity, verdict

_RANK = {verdict.PASS: 0, verdict.MARGINAL: 1, verdict.FAIL: 2}


def _verdicts(discretionary_income, buffer_pct, residual_floor_amount, instalments):
    cap, _binding = capacity.max_affordable_instalment_before_overlay(
        discretionary_income, buffer_pct, residual_floor_amount)
    exe = Engine().bind(verdict.affordability_verdict_code, mode="interpreted")
    return [
        exe.score({"evidence_sufficiency_code": verdict.EVIDENCE_OK, "max_affordable_instalment": cap,
                   "proposed_instalment": i})["affordability_verdict_code"]
        for i in instalments
    ]


def _is_monotone(verdicts) -> bool:
    ranks = [_RANK[v] for v in verdicts]
    return ranks == sorted(ranks)


def test_monotone_around_the_buffer_bound():
    """discretionary_income * (1 - buffer) < discretionary_income - floor: the buffer binds."""
    instalments = [1000.0, 4000.0, 6000.0, 6645.0, 6646.0, 7000.0, 12000.0]
    verdicts = _verdicts(8000.0, 0.20, 500.0, instalments)
    assert _is_monotone(verdicts), verdicts


def test_monotone_around_the_residual_floor_bound():
    """A small buffer and a large residual floor: the floor binds instead."""
    instalments = [1000.0, 3000.0, 5000.0, 5999.0, 6000.0, 6001.0, 10000.0]
    verdicts = _verdicts(8000.0, 0.05, 2000.0, instalments)
    assert _is_monotone(verdicts), verdicts


def test_monotone_at_the_exact_crossover_between_the_two_constraints():
    """Buffer and floor constraints are equal (§5.6.2's "the binding one wins" has no
    single winner) -- the classic case a naive implementation gets wrong."""
    # discretionary=10000, buffer=0.20 -> from_buffer=8000; floor=2000 -> from_floor=8000: tied.
    instalments = [7500.0, 7999.0, 8000.0, 8001.0, 8400.0, 9000.0]
    verdicts = _verdicts(10000.0, 0.20, 2000.0, instalments)
    assert _is_monotone(verdicts), verdicts


def test_monotone_with_zero_or_negative_discretionary_income():
    """Capacity floors at zero (§5.6.1: discretionary income itself is never floored, but
    capacity cannot be negative) -- every positive instalment must fail or be marginal."""
    instalments = [0.0, 1.0, 500.0, 5000.0]
    verdicts = _verdicts(-2000.0, 0.20, 500.0, instalments)
    assert _is_monotone(verdicts), verdicts


def test_monotone_across_a_fine_sweep_of_instalments_and_parameter_combinations():
    """A broad sweep across several (buffer, floor, discretionary income) combinations --
    not merely asserted at one point (acceptance §10 item 3)."""
    combos = [(0.10, 300.0), (0.25, 800.0), (0.35, 1_700.0), (0.60, 2_300.0)]
    instalments = [i * 250.0 for i in range(0, 60)]
    for di in (2_000.0, 6_000.0, 15_000.0):
        for buffer_pct, floor in combos:
            verdicts = _verdicts(di, buffer_pct, floor, instalments)
            assert _is_monotone(verdicts), (di, buffer_pct, floor, verdicts)
