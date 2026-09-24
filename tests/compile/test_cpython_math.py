"""`round(x, n)` inside kernels gives CPython's answers, so every mode agrees."""
import math
import random

import polars as pl
from numba import njit

from decider import flow
from decider.engine import Engine
from decider.engine.compile.cpython import cpython_round

# Decimal halves: the double sits just below or above the .5 (or on it, 0.125),
# and numba's scale-round-unscale gets some of them wrong (2.675, 1234.565).
HALVES = [2.675, 0.125, 0.375, 1.005, 1.015, 8.345, 0.285, 1234.565, -2.675, -0.125, -0.001]


@njit
def _kernel_round(x, n):
    return round(x, n)


def cents(amount: float) -> float:
    return round(amount, 2)


def test_known_disagreeing_halves_round_like_cpython_in_every_mode():
    frame = pl.DataFrame({"amount": HALVES})
    want = [round(x, 2) for x in HALVES]
    for mode in ("interpreted", "stepped", "fused"):
        assert Engine().bind(flow(cents), mode=mode).run(frame)["cents"].to_list() == want, mode


def test_kernel_round_matches_cpython_on_random_values_and_digits():
    rng = random.Random(7)
    for _ in range(20_000):
        n = rng.randint(-6, 10)
        x = rng.choice([rng.uniform(-1e6, 1e6), round(rng.uniform(-1e4, 1e4), rng.randint(0, 6)),
                        rng.uniform(-1, 1) * 10 ** rng.randint(-20, 20)])
        got, want = _kernel_round(x, n), round(x, n)
        assert got == want and math.copysign(1, got) == math.copysign(1, want), (x, n, got, want)


def test_edge_values_round_like_cpython():
    for x, n in [(2.5, 0), (-2.5, 0), (-0.0, 2), (1250.0, -2), (1350.0, -2), (-1250.0, -2),
                 (7.0621925386166184e16, -1), (495172.47519245883, 10), (1.7e308, 2), (math.inf, 2)]:
        assert _kernel_round(x, n) == round(x, n) == cpython_round(x, n), (x, n)
    assert math.isnan(_kernel_round(math.nan, 2))


@njit
def _kernel_pow(x, n):
    return x ** n


def annual_rate(monthly: float) -> float:
    return (1.0 + monthly) ** 12 - 1.0


def test_float_to_an_int_power_matches_cpython_in_every_mode():
    rng = random.Random(3)
    for _ in range(5_000):
        x, n = rng.uniform(-3, 3), rng.randint(-40, 400)
        assert _kernel_pow(x, n) == x ** n, (x, n)
    frame = pl.DataFrame({"monthly": [0.3 / 12, 0.2675 / 12, 0.0123]})
    want = [annual_rate(x) for x in frame["monthly"]]
    for mode in ("interpreted", "stepped", "fused"):
        assert Engine().bind(flow(annual_rate), mode=mode).run(frame)["annual_rate"].to_list() == want, mode
