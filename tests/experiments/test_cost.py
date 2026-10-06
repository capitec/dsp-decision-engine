from __future__ import annotations

from decider import flow
from decider.lab.experiments._cost import estimate_cost


def test_cost_counts_rows_times_calls():
    def a(x: float) -> float:
        return x + 1

    def b(a: float) -> float:
        return a * 2

    cost = estimate_cost(flow(a, b, name="f"), 10)
    assert (cost.rows, cost.calls, cost.row_operations) == (10, 2, 20)
