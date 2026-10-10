import decimal as stdlib
import operator
import random
from functools import lru_cache

import pytest
from numba import njit

import decikit._engine.compile.njit  # noqa: F401  (registers the decimal type and overloads)
from decikit._engine.compile.semantics.decimals import raw, wrapper
from decikit.ops import decimal
from decikit.ops.decimal import Decimal, Rounding

CTX = stdlib.Context(prec=200)
HALF_EVEN = stdlib.ROUND_HALF_EVEN
MODE = {Rounding.HALF_UP: stdlib.ROUND_HALF_UP, Rounding.HALF_EVEN: HALF_EVEN, Rounding.DOWN: stdlib.ROUND_DOWN}
INT64_MAX, INT64_MIN = 2 ** 63 - 1, -2 ** 63


def wrap64(n: int) -> int:
    return (n + 2 ** 63) % 2 ** 64 - 2 ** 63


def fits(n: int) -> bool:
    return INT64_MIN <= n <= INT64_MAX


def value(r: int, s: int) -> stdlib.Decimal:
    return stdlib.Decimal(r).scaleb(-s, CTX)


def raw_at(x: stdlib.Decimal, s: int) -> int:
    return int(x.scaleb(s, CTX).to_integral_value(HALF_EVEN))


def edges(scale: int) -> list[int]:
    p = 10 ** scale
    base = {0, 1, 2, 5, 10, 15, 25, 35, 49, 50, 51, 99, 100, 101, 125, 150, 250, 375, 5000, 9999, 12345, 10 ** 6 + 5, 10 ** 9,
            10 ** 12 + 5, 2 ** 31, 2 ** 32, 2 ** 32 + 1, 2 ** 53 - 1, 2 ** 53, 2 ** 53 + 1, 2 ** 62, INT64_MAX // 3,
            INT64_MAX - 1, INT64_MAX, p - 1, p, p + 1, p // 2, p // 2 + 1, 3 * p // 2, 2 * p - 1, 5 * p + p // 2}
    return sorted(x for x in {*base, *(-b for b in base), INT64_MIN} if fits(x))


PAIRS = [(2, 2), (2, 4), (4, 2), (4, 4), (4, 0), (0, 4), (0, 18), (18, 0), (18, 18), (6, 3)]
rng = random.Random(7)


def random_raws(n: int) -> list[int]:
    return [rng.choice((-1, 1)) * rng.getrandbits(rng.randint(1, 63)) for _ in range(n)]


@lru_cache(maxsize=None)
def kernels(sa: int, sb: int):
    wa, wb = wrapper(sa), wrapper(sb)

    @njit(nogil=True)
    def many(x, y):
        a, b = wa(x), wb(y)
        return (raw(a + b), raw(a - b), raw(a * b), a < b, a <= b, a == b, a != b, a > b, a >= b, raw(max(a, b)), raw(min(a, b)))

    @njit(nogil=True)
    def divide(x, y):
        return raw(wa(x) / wb(y))

    return many, divide


def expected(a: int, sa: int, b: int, sb: int) -> dict:
    s = max(sa, sb)
    x, y = value(a, sa), value(b, sb)
    out = {"add": raw_at(CTX.add(x, y), s), "sub": raw_at(CTX.subtract(x, y), s), "mul": raw_at(CTX.multiply(x, y), s),
           "lt": x < y, "le": x <= y, "eq": x == y, "ne": x != y, "gt": x > y, "ge": x >= y,
           "max": raw_at(max(x, y), s), "min": raw_at(min(x, y), s)}
    if b:
        out["div"] = raw_at(CTX.divide(x, y), s)
    return out


def check_pair(a: int, sa: int, b: int, sb: int) -> None:
    many, divide = kernels(sa, sb)
    want = expected(a, sa, b, sb)
    got = many(a, b)
    names = ("add", "sub", "mul", "lt", "le", "eq", "ne", "gt", "ge", "max", "min")
    for name, g in zip(names, got):
        w = want[name]
        assert g == (wrap64(w) if name in ("add", "sub", "mul", "max", "min") else w), (name, a, sa, b, sb)
    x, y = Decimal.from_raw(a, sa), Decimal.from_raw(b, sb)
    s = max(sa, sb)
    py = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv}
    if b:
        py_names = ("add", "sub", "mul", "div")
        assert divide(a, b) == wrap64(want["div"]), ("div", a, sa, b, sb)
    else:
        py_names = ("add", "sub", "mul")
        with pytest.raises(ZeroDivisionError):
            divide(a, b)
        with pytest.raises(ZeroDivisionError):
            x / y
    for name in py_names:
        if fits(want[name]):
            r = py[name](x, y)
            assert (r.raw, r.scale) == (want[name], s), (name, a, sa, b, sb)
        else:
            with pytest.raises(OverflowError):
                py[name](x, y)
    for name, op in (("lt", operator.lt), ("le", operator.le), ("eq", operator.eq), ("ne", operator.ne), ("gt", operator.gt), ("ge", operator.ge)):
        assert op(x, y) is want[name], (name, a, sa, b, sb)
    assert max(x, y) == Decimal.from_raw(want["max"], s) if fits(want["max"]) else True


@pytest.mark.parametrize("sa,sb", PAIRS)
def test_decimal_operators_equal_stdlib_and_python_over_the_edge_corpus(sa, sb):
    for a in edges(sa):
        for b in edges(sb):
            check_pair(a, sa, b, sb)


@pytest.mark.parametrize("sa,sb", PAIRS)
def test_decimal_operators_equal_stdlib_and_python_on_random_operands(sa, sb):
    for a, b in zip(random_raws(1500), random_raws(1500)):
        check_pair(a, sa, b, sb)


def test_division_chunks_for_wide_scale_differences_match_stdlib():
    # scale 0 over scale 18 needs a 10**36 factor, past 128 bits in one step
    many, divide = kernels(0, 18)
    for a in edges(0)[::3]:
        for b in edges(18)[::3]:
            if b:
                assert divide(a, b) == wrap64(raw_at(CTX.divide(value(a, 0), value(b, 18)), 18)), (a, b)


@pytest.mark.parametrize("sa,sb", [(2, 2), (2, 4), (4, 2), (1, 1), (0, 2), (2, 0), (3, 1)])
def test_every_small_operand_pair_covers_every_tie(sa, sb):
    for a in range(-30, 31):
        for b in range(-30, 31):
            check_pair(a, sa, b, sb)


@lru_cache(maxsize=None)
def int_kernels(sa: int):
    wa = wrapper(sa)

    @njit(nogil=True)
    def right(x, i):
        a = wa(x)
        return (raw(a + i), raw(a - i), raw(a * i), a < i, a <= i, a == i, a != i, a > i, a >= i)

    @njit(nogil=True)
    def left(x, i):
        a = wa(x)
        return (raw(i + a), raw(i - a), raw(i * a), i < a, i <= a, i == a, i != a, i > a, i >= a)

    @njit(nogil=True)
    def divide(x, i):
        return raw(wa(x) / i)

    return right, left, divide


INTS = sorted({0, 1, -1, 2, 3, 7, -7, 10, 100, 12345, 10 ** 6, -10 ** 6, 10 ** 12, 2 ** 31, 2 ** 53, 2 ** 53 + 1, 10 ** 17, 10 ** 18,
               -10 ** 18, INT64_MAX, INT64_MIN, INT64_MAX - 1})


@pytest.mark.parametrize("sa", [0, 2, 4, 18])
def test_int_operands_act_as_whole_units_on_either_side(sa):
    right, left, divide = int_kernels(sa)
    for a in edges(sa):
        for i in INTS:
            for k, flip in ((right, False), (left, True)):
                x, y = (i, a) if flip else (a, i)
                sx, sy = (0, sa) if flip else (sa, 0)
                want = expected(x, sx, y, sy)
                got = k(a, i)
                for g, name in zip(got, ("add", "sub", "mul", "lt", "le", "eq", "ne", "gt", "ge")):
                    w = want[name]
                    assert g == (wrap64(w) if name in ("add", "sub", "mul") else w), (name, a, sa, i, flip)
                d = Decimal.from_raw(a, sa)
                for name, op in (("add", operator.add), ("sub", operator.sub), ("mul", operator.mul)):
                    if fits(want[name]):
                        r = op(i, d) if flip else op(d, i)
                        assert (r.raw, r.scale) == (want[name], sa)
                    else:
                        with pytest.raises(OverflowError):
                            op(i, d) if flip else op(d, i)
            if i:
                w = raw_at(CTX.divide(value(a, sa), value(i, 0)), sa)
                assert divide(a, i) == wrap64(w), (a, sa, i)
                if fits(w):
                    assert Decimal.from_raw(a, sa) / i == Decimal.from_raw(w, sa)
            else:
                with pytest.raises(ZeroDivisionError):
                    divide(a, i)


FLOATS = [0.0, -0.0, 0.5, -0.5, 1.5, 2.5, 3.0, -2.0, 0.1, 0.3, 0.0175, 1 / 3, 2.675, 0.125, 1e-4, 1e-9, 12345.6789, 1e6, 2.0 ** -30,
          2.0 ** 40, -1e5]


@lru_cache(maxsize=None)
def float_kernels(sa: int):
    wa = wrapper(sa)

    @njit(nogil=True)
    def k(x, f):
        a = wa(x)
        return (raw(a * f), raw(f * a), a < f, a <= f, a == f, a != f, a > f, a >= f, f < a, f >= a)

    @njit(nogil=True)
    def divide(x, f):
        return raw(wa(x) / f)

    return k, divide


def exact_round(x: float) -> int:
    return int(stdlib.Decimal(x).to_integral_value(HALF_EVEN))


@pytest.mark.parametrize("sa", [0, 2, 4])
def test_float_operands_multiply_divide_and_compare_as_documented(sa):
    k, divide = float_kernels(sa)
    for a in edges(sa):
        d = Decimal.from_raw(a, sa)
        for f in FLOATS:
            as_float = float(a) / float(10 ** sa)
            got = k(a, f)
            assert got[2:] == (as_float < f, as_float <= f, as_float == f, as_float != f, as_float > f, as_float >= f, f < as_float, f >= as_float), (a, sa, f)
            assert (d < f, d <= f, d == f, d != f, d > f, d >= f, f < d, f >= d) == got[2:]
            product = float(a) * f
            if abs(product) < 2.0 ** 62:   # a compiled step's float to int64 conversion is undefined beyond int64
                assert got[0] == got[1] == exact_round(product), (a, sa, f)
                assert (d * f).raw == (f * d).raw == exact_round(product) and (d * f).scale == sa
            elif not fits(exact_round(product)):
                with pytest.raises(OverflowError):
                    d * f
            if f:
                quotient = float(a) / f
                if abs(quotient) < 2.0 ** 62:
                    assert divide(a, f) == exact_round(quotient), (a, sa, f)
                    assert (d / f).raw == exact_round(quotient)
                elif not fits(exact_round(quotient)):
                    with pytest.raises(OverflowError):
                        d / f
            else:
                with pytest.raises(ZeroDivisionError):
                    divide(a, f)
                with pytest.raises(ZeroDivisionError):
                    d / f


def test_float_comparison_is_the_same_expression_in_python_and_kernel_past_two_to_the_53():
    k, _ = float_kernels(4)
    for a in (2 ** 53 + 1, 2 ** 53 + 3, 2 ** 60 + 1, INT64_MAX):
        f = float(a) / 10 ** 4
        assert k(a, f)[4] is True and Decimal.from_raw(a, 4) == f


@lru_cache(maxsize=None)
def unary_kernel(s: int):
    w = wrapper(s)

    @njit(nogil=True)
    def k(x):
        a = w(x)
        return raw(-a), raw(abs(a)), float(a), int(a)

    return k


@pytest.mark.parametrize("s", [0, 2, 4, 18])
def test_negation_abs_float_and_int_agree_and_int64_min_wraps(s):
    k = unary_kernel(s)
    for a in edges(s):
        neg, ab, f, i = k(a)
        assert neg == wrap64(-a) and ab == wrap64(abs(a))
        d = Decimal.from_raw(a, s)
        assert f == float(d) == float(a) / float(10 ** s) and i == int(d) == int(value(a, s))
        if a == INT64_MIN:
            with pytest.raises(OverflowError):
                -d
            with pytest.raises(OverflowError):
                abs(d)
        else:
            assert (-d).raw == neg and abs(d).raw == ab


# ---- round

@lru_cache(maxsize=None)
def round_kernel(s: int):
    w = wrapper(s)

    @njit(nogil=True)
    def k(x, digits, mode):
        return raw(decimal.round(w(x), digits, mode))

    return k


@pytest.mark.parametrize("mode", list(Rounding))
@pytest.mark.parametrize("s", [0, 2, 4, 18])
def test_round_in_a_kernel_matches_stdlib_python_and_every_mode(mode, s):
    k = round_kernel(s)
    for digits in range(-21, s + 2):
        for a in edges(s)[::2]:
            want = int(value(a, s).quantize(stdlib.Decimal(1).scaleb(-digits), MODE[mode], CTX).scaleb(s, CTX))
            assert k(a, digits, mode) == wrap64(want), (a, s, digits, mode)
            if fits(want):
                assert decimal.round(Decimal.from_raw(a, s), digits, mode).raw == want
            else:
                with pytest.raises(OverflowError):
                    decimal.round(Decimal.from_raw(a, s), digits, mode)


def test_round_negatives_near_zero_do_not_round_to_minus_one():
    k = round_kernel(2)
    want = {Rounding.HALF_UP: [0, -100, -100, 0, 0], Rounding.HALF_EVEN: [0, 0, -100, 0, 0], Rounding.DOWN: [0, 0, 0, 0, 0]}
    for mode in Rounding:
        assert [k(r, 0, mode) for r in (-49, -50, -51, -1, 0)] == want[mode]


def test_round_with_default_arguments_and_on_floats_in_a_kernel():
    w = wrapper(2)

    @njit
    def f(x, y):
        return raw(decimal.round(w(x))), raw(decimal.round(w(x), 1)), decimal.round(y), decimal.round(y, 1, Rounding.DOWN)

    assert f(250, 2.55) == (300, 250, 3.0, 2.5)
