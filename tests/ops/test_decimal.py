import decimal as stdlib
import math
import random

import numpy as np
import polars as pl
import pytest
from numba import njit

from decider import flow
from decider.ops import decimal
from decider.ops.decimal import Rounding
from decider.testing import assert_equivalent

REFERENCE = {Rounding.HALF_UP: stdlib.ROUND_HALF_UP, Rounding.HALF_EVEN: stdlib.ROUND_HALF_EVEN,
             Rounding.DOWN: stdlib.ROUND_DOWN}
CONTEXT = stdlib.Context(prec=400)


def reference(x: float, digits: int, mode: Rounding) -> float:
    return float(stdlib.Decimal(x).quantize(stdlib.Decimal(1).scaleb(-digits), REFERENCE[mode], CONTEXT))


@njit
def kernel_round(x, digits, mode):
    return decimal.round(x, digits, mode)


def same(got: float, want: float) -> bool:
    return got == want and math.copysign(1, got) == math.copysign(1, want)


TIES = [k / 8 for k in range(-80, 81)] + [2.675, 1.005, 1234.565, 0.285, -2.675, 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5]


@pytest.mark.parametrize("mode", list(Rounding))
@pytest.mark.parametrize("digits", [0, 1, 2, 3, 4, -1, -2])
def test_every_mode_matches_decimal_on_the_exact_double_for_ties(mode, digits):
    for x in TIES + [t * 10.0 ** -digits for t in TIES[:40]]:
        want = reference(x, digits, mode)
        assert same(decimal.round(x, digits, mode), want), (x, digits, mode)
        assert same(kernel_round(x, digits, mode), want), (x, digits, mode)


@pytest.mark.parametrize("mode", list(Rounding))
def test_every_mode_matches_decimal_on_random_values(mode):
    rng = random.Random(11)
    for _ in range(3000):
        digits = rng.randint(-3, 6)
        x = rng.choice([rng.uniform(-1e6, 1e6), rng.uniform(-1, 1), rng.uniform(-1e15, 1e15),
                        round(rng.uniform(-1e4, 1e4), rng.randint(0, 6))])
        assert same(decimal.round(x, digits, mode), reference(x, digits, mode)), (x, digits, mode)


def test_the_exact_double_is_rounded_so_2_675_gives_2_67_in_every_mode():
    assert [decimal.round(2.675, 2, mode) for mode in Rounding] == [2.67] * 3


def test_default_is_half_up_with_zero_digits():
    assert (decimal.round(2.5), decimal.round(-2.5), decimal.round(3.5)) == (3.0, -3.0, 4.0)


def test_modes_differ_on_ties_and_down_truncates_toward_zero():
    assert [decimal.round(2.5, mode=m) for m in Rounding] == [3.0, 2.0, 2.0]
    assert [decimal.round(-2.5, mode=m) for m in Rounding] == [-3.0, -2.0, -2.0]
    assert decimal.round(-2.9, mode=Rounding.DOWN) == -2.0
    assert decimal.round(2.9, mode=Rounding.DOWN) == 2.0


@pytest.mark.parametrize("mode", list(Rounding))
def test_values_near_2_pow_53_round_like_decimal(mode):
    p = 2.0 ** 53
    for x in [p, p - 1, p - 2, p + 2, 2.0 ** 52 + 0.5, 2.0 ** 52 - 0.5, 2.0 ** 51 + 0.25, -(p - 1), -(2.0 ** 52 + 0.5),
              np.nextafter(p, 0.0), np.nextafter(2.0 ** 52, 0.0)]:
        for digits in (0, 1, -1):
            assert same(decimal.round(float(x), digits, mode), reference(float(x), digits, mode)), (x, digits, mode)


@pytest.mark.parametrize("mode", list(Rounding))
def test_zero_keeps_its_sign_and_nan_and_infinities_pass_through(mode):
    assert math.copysign(1, decimal.round(-0.0, 2, mode)) == -1
    assert math.copysign(1, decimal.round(0.0, 2, mode)) == 1
    assert math.copysign(1, decimal.round(-0.0004, 3, mode)) == -1
    assert math.isnan(decimal.round(math.nan, 2, mode)) and math.isnan(kernel_round(math.nan, 2, mode))
    assert decimal.round(math.inf, 2, mode) == math.inf and decimal.round(-math.inf, 0, mode) == -math.inf
    assert kernel_round(-math.inf, 1, mode) == -math.inf


@pytest.mark.parametrize("mode", list(Rounding))
def test_huge_digits_and_huge_values_do_not_overflow(mode):
    assert decimal.round(1e300, 4, mode) == 1e300
    assert decimal.round(123.456, -30, mode) == 0.0
    assert decimal.round(1.25, 30, mode) == 1.25


def step(x: float) -> float:
    return decimal.round(x, 2, Rounding.HALF_EVEN) + decimal.round(x, 0) * 1000.0


def test_a_step_calling_round_agrees_in_every_mode():
    xs = [*TIES, math.nan, math.inf, -0.0, 2.0 ** 53, 1e18]
    out = assert_equivalent(flow(step, name="p"), pl.DataFrame({"x": xs}))
    assert out["step"].to_list()[:5] == pytest.approx([decimal.round(x, 2, Rounding.HALF_EVEN) + decimal.round(x) * 1000.0
                                                       for x in xs[:5]])
