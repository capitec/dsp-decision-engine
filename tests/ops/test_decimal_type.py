import copy
import decimal as stdlib
import pickle

import pytest

from decikit.ops import decimal
from decikit.ops.decimal import Currency, Decimal, Rounding

CTX = stdlib.Context(prec=200)
MODE = {Rounding.HALF_UP: stdlib.ROUND_HALF_UP, Rounding.HALF_EVEN: stdlib.ROUND_HALF_EVEN, Rounding.DOWN: stdlib.ROUND_DOWN}


def exact(d: Decimal) -> stdlib.Decimal:
    return stdlib.Decimal(d.raw).scaleb(-d.scale, CTX)


# ---- construction

def test_int_defaults_to_scale_zero():
    d = Decimal(3)
    assert (d.raw, d.scale) == (3, 0)


def test_int_with_scale_is_whole_units():
    assert Decimal(3, 2).raw == 300


def test_str_scale_is_the_digits_after_the_point():
    assert (Decimal("2.65").raw, Decimal("2.65").scale) == (265, 2)
    assert (Decimal("2.650").raw, Decimal("2.650").scale) == (2650, 3)
    assert (Decimal("-0.5").raw, Decimal("-0.5").scale) == (-5, 1)
    assert (Decimal("1e3").raw, Decimal("1e3").scale) == (1000, 0)


@pytest.mark.parametrize("text,scale,raw", [("2.675", 2, 268), ("2.665", 2, 266), ("-2.675", 2, -268), ("0.125", 2, 12)])
def test_str_rounds_to_the_scale_half_even_from_its_digits(text, scale, raw):
    assert Decimal(text, scale).raw == raw


@pytest.mark.parametrize("value,scale,raw", [
    (2.675, 2, 267),     # the double is 2.67499999999999982...
    (0.125, 2, 12),      # an exact tie goes to the even neighbour
    (0.375, 2, 38),
    (2.65, 4, 26500),    # 2.6499999999999999 rounds to the scale, not down
    (-2.675, 2, -267),
    (1e-9, 2, 0),
])
def test_float_rounds_to_the_scale_from_its_exact_value_without_a_warning(value, scale, raw, recwarn):
    assert Decimal(value, scale).raw == raw
    assert not recwarn


def test_float_without_a_scale_is_a_value_error():
    with pytest.raises(ValueError, match="needs a scale"):
        Decimal(2.5)


@pytest.mark.parametrize("bad", [-1, 19, 2.0, "2"])
def test_scale_outside_zero_to_eighteen_is_a_value_error(bad):
    with pytest.raises(ValueError, match="scale"):
        Decimal(1, bad)


@pytest.mark.parametrize("bad", ["abc", "", "1/3", "nan", "inf"])
def test_non_numeric_str_is_a_value_error(bad):
    with pytest.raises(ValueError):
        Decimal(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_float_is_a_value_error(bad):
    with pytest.raises(ValueError, match="finite"):
        Decimal(bad, 2)


@pytest.mark.parametrize("bad", [None, [1], (1, 2), b"1", 1j])
def test_other_types_are_a_type_error(bad):
    with pytest.raises(TypeError):
        Decimal(bad)


@pytest.mark.parametrize("args", [(2 ** 63, 0), (2 ** 63 // 100 + 1, 2), ("92233720368547758.08", 2), (1e19, 0)])
def test_value_outside_int64_is_an_overflow_error(args):
    with pytest.raises(OverflowError):
        Decimal(*args)


def test_a_decimal_or_a_stdlib_decimal_is_read_by_the_construction_rules():
    assert Decimal(stdlib.Decimal("2.675")).raw == 2675 and Decimal(stdlib.Decimal("2.675")).scale == 3
    assert Decimal(stdlib.Decimal("2.675"), 2).raw == 268 and Decimal(Decimal("2.675"), 2).raw == 268
    assert Currency(Decimal("2.5", units="R")).units == "R" and Currency(stdlib.Decimal("2.5")).raw == 25000
    with pytest.raises(ValueError, match="finite"):
        Decimal(stdlib.Decimal("NaN"))


def test_int64_bounds_are_accepted():
    assert Decimal(2 ** 63 - 1).raw == 2 ** 63 - 1
    assert Decimal(-2 ** 63).raw == -2 ** 63


def test_decimal_is_not_an_int():
    assert not isinstance(Decimal(1), int)
    assert not isinstance(Currency(1), int)


def test_decimal_is_immutable():
    d = Decimal(1)
    for name in ("raw", "scale", "units", "other"):
        with pytest.raises(AttributeError):
            setattr(d, name, 2)


def test_from_raw_takes_the_scaled_int():
    assert Decimal.from_raw(265, 2) == Decimal("2.65")
    with pytest.raises(OverflowError):
        Decimal.from_raw(2 ** 63, 2)


def test_pickle_and_copy_round_trip():
    for d in (Decimal("2.50", units="R"), Currency(2, 6500), Decimal(-7)):
        for twin in (pickle.loads(pickle.dumps(d)), copy.copy(d), copy.deepcopy(d)):
            assert (twin.raw, twin.scale, twin.units, type(twin)) == (d.raw, d.scale, d.units, type(d))


# ---- Currency

def test_currency_is_scale_four():
    assert (Currency(2.65).raw, Currency(2.65).scale) == (26500, 4)
    assert Currency(3).raw == 30000


@pytest.mark.parametrize("major,minor,raw", [
    (2, 6, 20006), (2, 6500, 26500), (0, 5, 5), (-2, 5, -20005), (-2, 6500, -26500), (0, 0, 0), (2, 9999, 29999),
    (-1, 0, -10000),
])
def test_currency_major_minor_is_exact_and_the_sign_applies_to_the_whole(major, minor, raw):
    assert Currency(major, minor).raw == raw


def test_negative_under_one_unit_is_the_negation():
    assert (-Currency(0, 5)).raw == -5


@pytest.mark.parametrize("minor", [-1, 10000, 12345])
def test_currency_minor_outside_its_range_is_a_value_error(minor):
    with pytest.raises(ValueError, match="minor"):
        Currency(2, minor)


@pytest.mark.parametrize("args", [(2.5, 5), (2, 5.0), ("2", 5), (2, "5")])
def test_currency_major_minor_takes_ints_only(args):
    with pytest.raises(TypeError):
        Currency(*args)


def test_currency_beyond_int64_is_an_overflow_error():
    with pytest.raises(OverflowError):
        Currency(922337203685478, 0)
    assert Currency(922337203685477, 5807).raw == 2 ** 63 - 1


def test_currency_from_str_rounds_half_even_at_four_places():
    assert Currency("2.00005").raw == 20000
    assert Currency("2.00015").raw == 20002


# ---- repr and units

def test_repr_and_str():
    assert repr(Decimal("2.50")) == "Decimal('2.50')" and str(Decimal("2.50")) == "2.50"
    assert repr(Currency(2, 6500)) == "Currency('2.6500')"
    assert repr(Decimal(-7)) == "Decimal('-7')"
    assert str(-Currency(0, 5)) == "-0.0005"
    assert str(Decimal("0.05")) == "0.05"


def test_units_come_before_the_amount_in_repr_and_str():
    assert repr(Currency(2, 6500, units="R")) == "R 2.6500"
    assert str(Currency(-2, 6500, units="ZAR")) == "ZAR -2.6500"
    assert repr(Decimal(5, units="$")) == "$ 5"


def test_units_do_not_enter_equality_or_hash():
    assert Currency(2, units="R") == Currency(2, units="$") == Currency(2)
    assert len({Currency(2, units="R"), Currency(2, units="$"), Decimal(2)}) == 1


def test_units_survive_arithmetic_when_they_agree_and_drop_when_they_do_not():
    r, usd = Currency(2, units="R"), Currency(3, units="$")
    assert (r + r).units == "R" and (r * 2).units == "R" and (r + 1).units == "R" and (-r).units == "R"
    assert (r + usd).units is None and (r + Currency(3)).units is None
    assert (1.5 * r).units == "R" and abs(r).units == "R"


def test_arithmetic_on_a_currency_returns_a_plain_decimal_but_negation_and_abs_stay_a_currency():
    assert type(Currency(1) + Currency(1)) is Decimal
    assert type(-Currency(1)) is Currency and type(abs(Currency(-1))) is Currency
    assert repr(-Currency(2, 6500, units="R")) == "R -2.6500"


# ---- equality, ordering and hash

def test_equality_ignores_the_scale():
    assert Decimal("2.5") == Decimal("2.5000") == Decimal(2.5, 3)
    assert hash(Decimal("2.5")) == hash(Decimal("2.5000"))


def test_equality_with_int_and_float():
    assert Decimal(3) == 3 and Decimal(3, 4) == 3 and Decimal("2.5") != 2
    assert Decimal("0.30") == 0.3                      # compared as float(raw) / 10**scale
    assert Decimal("0.30") != 0.1 + 0.2
    assert hash(Decimal(5)) == hash(5)


def test_ordering_across_scales_ints_and_floats():
    assert Decimal("2.5") < Decimal("2.51") <= Decimal("2.510") < 3 < Decimal("3.0001")
    assert Decimal("2.5") > 2 and Decimal("2.5") >= 2.5 and Decimal("2.5") < 2.6 and 2.4 < Decimal("2.5")
    assert not Decimal("2.5") < float("nan") and Decimal("2.5") != float("nan")


def test_comparison_is_exact_near_int64_bounds():
    assert Decimal.from_raw(2 ** 63 - 1, 4) > Decimal.from_raw(2 ** 63 - 2, 4)
    assert Decimal.from_raw(2 ** 63 - 1, 0) > Decimal.from_raw(2 ** 63 // 10 ** 18, 18)


def test_comparison_with_other_types_is_not_implemented():
    assert Decimal(1) != "1" and Decimal(1) != None  # noqa: E711
    with pytest.raises(TypeError):
        Decimal(1) < "1"


def test_comparison_with_a_stdlib_decimal_is_exact():
    assert Decimal("2.50") == stdlib.Decimal("2.5") and Decimal("2.5") < stdlib.Decimal("2.51")
    assert Decimal("2.5") != stdlib.Decimal("NaN") and not Decimal("2.5") < stdlib.Decimal("NaN")


def test_decimal_is_hashable_and_a_dict_key():
    assert {Decimal("1.50"): 1}[Decimal("1.5")] == 1


# ---- arithmetic against stdlib decimal

HALF_EVEN = stdlib.ROUND_HALF_EVEN


def quantum(scale: int) -> stdlib.Decimal:
    return stdlib.Decimal(1).scaleb(-scale)


@pytest.mark.parametrize("a,b", [("0.10", "0.05"), ("0.15", "0.1"), ("2.5", "0.5"), ("-0.10", "0.05"), ("1.5", "0.25"),
                                 ("2.50", "0.0005"), ("0.0001", "0.5"), ("-2.5", "-0.5")])
def test_ops_match_stdlib_decimal_half_even_at_the_larger_scale(a, b):
    x, y = Decimal(a), Decimal(b)
    s = max(x.scale, y.scale)
    q = quantum(s)
    for got, want in [(x + y, exact(x) + exact(y)), (x - y, exact(x) - exact(y)), (x * y, exact(x) * exact(y)),
                      (x / y, exact(x) / exact(y))]:
        assert got.scale == s
        assert exact(got) == want.quantize(q, HALF_EVEN, CTX), (a, b)


def test_mul_and_div_are_half_even_not_half_up():
    assert Decimal("0.5") * Decimal("0.5") == Decimal("0.2")       # 0.25
    assert Decimal("1.5") * Decimal("0.5") == Decimal("0.8")       # 0.75
    assert Decimal("2.5") * Decimal("0.5") == Decimal("1.2")       # 1.25
    assert Decimal("0.50") / Decimal("4") == Decimal("0.12")       # 0.125
    assert Decimal("1.50") / Decimal("4") == Decimal("0.38")       # 0.375
    assert Decimal("-1.50") / Decimal("4") == Decimal("-0.38")
    assert (Decimal("0.10") * Decimal("0.05")).raw == 0 and (Decimal("0.15") * Decimal("0.1")).raw == 2


def test_mixed_scale_results_take_the_larger_scale():
    assert (Decimal("3.1234") + Decimal("2.5")).scale == 4
    assert Decimal("3.1234") + Decimal("2.5") == Decimal("5.6234")
    assert Decimal("3.1234") * Decimal("2.5") == Decimal("7.8085")     # 7.8085 exactly
    assert Decimal("3.1234") / Decimal("2.5") == Decimal("1.2494")     # 1.24936


def test_an_int_is_whole_units_on_either_side():
    d = Decimal("2.50")
    assert d + 1 == 1 + d == Decimal("3.50") and d - 1 == Decimal("1.50") and 1 - d == Decimal("-1.50")
    assert d * 3 == 3 * d == Decimal("7.50") and d / 4 == Decimal("0.62")      # 0.625
    assert (d + 1).scale == (1 + d).scale == 2


def test_float_multiplier_gives_a_decimal_at_the_decimals_scale_half_even():
    assert Decimal("0.31") * 1.5 == Decimal("0.46")                    # 0.465 is the double 0.46499999...
    assert 1.5 * Currency(3) == Currency(4, 5000) and type(1.5 * Currency(3)) is Decimal
    assert (Decimal("0.05") * 0.5).raw == 2 and (Decimal("0.15") * 0.5).raw == 8       # 2.5 -> 2, 7.5 -> 8
    assert Decimal("2.50") / 0.5 == Decimal("5.00") and Decimal("0.05") / 2.0 == Decimal("0.02")


def test_unary_minus_abs_bool_and_conversions():
    d = Decimal("-2.75")
    assert -d == Decimal("2.75") and abs(d) == Decimal("2.75") and abs(Decimal("2.75")) == Decimal("2.75")
    assert float(d) == -2.75 and int(d) == -2 and int(Decimal("2.99")) == 2 and int(Decimal("-0.5")) == 0
    assert bool(Decimal(0)) is False and bool(Decimal("0.01")) is True


def test_sum_min_max_and_in_place_add():
    xs = [Decimal("0.10"), Decimal("0.2"), Decimal("0.3000")]
    assert sum(xs) == Decimal("0.6") and sum(xs, Decimal(0)) == Decimal("0.6") and sum(xs, Decimal(0)).scale == 4
    assert max(xs) == Decimal("0.3") and min(xs) == Decimal("0.1")
    total = Decimal(0)
    for x in xs:
        total += x
    assert total == Decimal("0.6")


def test_division_by_zero_raises():
    for zero in (Decimal(0), Decimal(0, 4), 0, 0.0):
        with pytest.raises(ZeroDivisionError):
            Decimal(1) / zero


# ---- the operations that are errors

def test_unsupported_operations_are_type_errors():
    d = Decimal("2.5")
    for bad in (lambda: d // 2, lambda: d % 2, lambda: d ** 2, lambda: 2 // d, lambda: 2 ** d, lambda: 2.0 / d,
                lambda: d + 1.5, lambda: 1.5 + d, lambda: d - 1.5, lambda: 1.5 - d, lambda: d + "1", lambda: d * d * "1",
                lambda: 2 / d, lambda: round(d), lambda: round(d, 1), lambda: divmod(d, 2)):
        with pytest.raises(TypeError):
            bad()


def test_float_over_decimal_says_to_use_float():
    with pytest.raises(TypeError, match=r"float\(d\)"):
        1.5 / Decimal(2)


def test_a_float_multiplier_does_not_leak_a_float():
    assert type(1.5 * Decimal(3)) is Decimal and type(Decimal(3) * 1.5) is Decimal


# ---- overflow raises in Python

@pytest.mark.parametrize("make", [
    lambda m: m + m, lambda m: m - (-m), lambda m: m * 2, lambda m: 2 * m, lambda m: -Decimal.from_raw(-2 ** 63, 0),
    lambda m: abs(Decimal.from_raw(-2 ** 63, 0)), lambda m: m * 1e19, lambda m: m + 1, lambda m: m * Decimal(2),
    lambda m: m / 0.1,
])
def test_a_result_outside_int64_raises_overflow_error(make):
    with pytest.raises(OverflowError):
        make(Decimal.from_raw(2 ** 63 - 1, 0))


def test_float_multiplier_that_is_not_a_number_raises():
    with pytest.raises(ValueError):
        Decimal(1) * float("nan")
    with pytest.raises(OverflowError):
        Decimal(1) * float("inf")


# ---- decimal.round

@pytest.mark.parametrize("mode", list(Rounding))
@pytest.mark.parametrize("scale,digits", [(2, 0), (2, 1), (2, 2), (2, 3), (4, 2), (4, 0), (4, -1), (4, -3), (4, -5), (3, 1), (0, -1), (0, 0)])
def test_round_matches_stdlib_in_every_mode_including_ties_and_negatives(mode, scale, digits):
    raws = {0, 1, 2, 5, 15, 25, 35, 45, 50, 55, 95, 99, 105, 150, 250, 350, 449, 450, 451, 500, 999, 1005, 12345, 99950, 100000,
            5 * 10 ** 5, 10 ** 12 + 5, 2 ** 53, 2 ** 62}
    for r in sorted(raws | {-x for x in raws}):
        d = Decimal.from_raw(r, scale)
        want = exact(d).quantize(stdlib.Decimal(1).scaleb(-digits), MODE[mode], CTX)
        got = decimal.round(d, digits, mode)
        assert got.scale == scale and exact(got) == want, (r, scale, digits, mode)


def test_round_default_mode_is_half_up_and_keeps_the_scale():
    assert decimal.round(Decimal("2.675"), 2) == Decimal("2.68") and decimal.round(Decimal("2.675"), 2).scale == 3
    assert decimal.round(Decimal("2.5")) == Decimal("3.0") and decimal.round(Decimal("-2.5")) == Decimal("-3.0")
    assert decimal.round(Decimal("2.5"), mode=Rounding.HALF_EVEN) == Decimal("2.0")
    assert decimal.round(Decimal("-2.99"), 1, Rounding.DOWN) == Decimal("-2.9")


def test_round_leaves_a_decimal_that_has_no_finer_digits_alone():
    d = Decimal("2.345")
    assert decimal.round(d, 3) is d and decimal.round(d, 9) is d


def test_round_on_a_float_is_unchanged():
    assert decimal.round(2.675, 2) == 2.67 and decimal.round(2.5) == 3.0 and decimal.round(2.5, 0, Rounding.HALF_EVEN) == 2.0


def test_round_up_past_int64_raises():
    with pytest.raises(OverflowError):
        decimal.round(Decimal.from_raw(2 ** 63 - 1, 0), -1)
