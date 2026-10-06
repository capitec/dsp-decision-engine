"""DCD101-110: each numeric check fires on its defect and stays quiet on the safe form."""
from typing import Annotated

import numpy as np
import pytest

from decider import flow, loop, param, step
from decider.lab import checks
from decider.lab.checks import Severity
from decider.ops import decimal
from decider.ops.decimal import Currency, Rounding


def run(*fns, alias):
    return [f for f in checks.run(flow(*fns, name="t"), [alias]).findings if f.check == alias]


def fires(fn, alias, n=1):
    found = run(fn, alias=alias)
    assert len(found) == n, found
    return found[0] if n == 1 else found


def quiet(fn, alias):
    assert run(fn, alias=alias) == []


# The user story: one flow, three defects, no false positive.
def inst(amount: Currency, rate: float = param(0.1)) -> Currency:
    return round(amount * rate / 12, 2)


def ratio(a: float, b: float) -> float:
    return a / b


def sq(cents: int) -> int:
    return cents ** 2


def test_money_story_reports_round_division_and_overflow_without_a_false_positive():
    report = checks.run(flow(inst, ratio, sq, name="t"))
    seen = {(f.code, f.step.path) for f in report.findings if f.code.startswith("DCD1")}
    assert seen == {("DCD103", "t/inst"), ("DCD104", "t/ratio"), ("DCD101", "t/sq")}
    assert next(f for f in report.findings if f.code == "DCD103").severity is Severity.ERROR


# DCD101
def times_months(amount: Currency, months: int) -> Currency:
    return amount * months


def times_float(amount: Currency, rate: float) -> Currency:
    return amount * rate


def constants() -> int:
    return 2 * 1024


def running_total(cents: int) -> int:
    total = 0
    for _ in range(3):
        total += cents
    return total


def test_a_decimal_product_with_an_int_is_an_error():
    finding = fires(times_months, "int64_overflow")
    assert finding.severity is Severity.ERROR and finding.line and finding.fix and "OverflowError" in finding.message


def test_plain_int_product_is_a_warning_and_float_or_constant_products_are_quiet():
    assert fires(sq, "int64_overflow").severity is Severity.WARNING
    quiet(times_float, "int64_overflow")
    quiet(constants, "int64_overflow")


def test_a_running_int_total_in_a_loop_is_flagged():
    assert fires(running_total, "int64_overflow").detail["expression"] == "total += cents"


# DCD102
def truncated(price: float) -> int:
    return int(price * 100)


def float_into_decimal(amount: Currency, rate: float) -> Currency:
    return float(amount) * rate


def test_int_times_100_truncation_is_flagged():
    assert fires(truncated, "float_money").detail["kind"] == "truncation"


def test_a_float_returned_into_a_decimal_output_is_flagged_but_a_decimal_times_a_float_is_not():
    assert fires(float_into_decimal, "float_money").detail["kind"] == "float_to_decimal"
    quiet(inst, "float_money")
    quiet(times_float, "float_money")


# DCD103
def rounded_plain(x: float) -> float:
    return round(x, 2)


def rounded_np(x: float) -> float:
    return np.round(x, 2)


def rounded_decimal(x: float) -> float:
    return decimal.round(x, 2, Rounding.HALF_UP)


def test_bare_round_is_a_warning_off_money_and_np_round_counts():
    assert fires(rounded_plain, "bare_round").severity is Severity.WARNING
    fires(rounded_np, "bare_round")


def test_the_decimal_helper_is_quiet():
    quiet(rounded_decimal, "bare_round")


# DCD104
def guarded_if(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def guarded_early(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def floored(a: float, b: float) -> float:
    return a / max(b, 1e-9)


def bounded_param(a: float, d: float = param(1.0, gt=0)) -> float:
    return a / d


def by_constant(a: float) -> float:
    return a / 12


def modulo(a: int, b: int) -> int:
    return a % b


def test_division_by_an_unproven_variable_fires_for_div_and_mod():
    fires(ratio, "division_by_variable")
    fires(modulo, "division_by_variable")


@pytest.mark.parametrize("fn", [guarded_if, guarded_early, floored, bounded_param, by_constant])
def test_division_that_is_provably_safe_is_quiet(fn):
    quiet(fn, "division_by_variable")


# DCD105
def ints_divided(a: int, b: int) -> float:
    return a / b


def test_true_division_of_ints_is_info_and_money_times_rate_over_12_is_quiet():
    assert fires(ints_divided, "int_true_division").severity is Severity.INFO
    quiet(inst, "int_true_division")


# DCD106
def grow(x: float, n: float) -> float:
    return x ** n


def exp_free(x: float) -> float:
    return np.exp(x)


def exp_clamped(x: float) -> float:
    return np.exp(min(x, 700.0))


def squared(x: float) -> float:
    return x ** 2


def test_unbounded_exponent_fires_and_constant_or_clamped_is_quiet():
    fires(grow, "pow_exp_overflow")
    fires(exp_free, "pow_exp_overflow")
    quiet(exp_clamped, "pow_exp_overflow")
    quiet(squared, "pow_exp_overflow")


# DCD107
def eq_const(x: float) -> bool:
    return x == 0.5


def eq_names(x: float, y: float) -> bool:
    return x == y


def eq_ints(x: int, y: int) -> bool:
    return x == y


def test_float_equality_fires_on_a_constant_or_two_floats_not_ints():
    fires(eq_const, "float_equality")
    fires(eq_names, "float_equality")
    quiet(eq_ints, "float_equality")


# DCD108
def root(x: float) -> float:
    return np.sqrt(x)


def root_of_square(x: float) -> float:
    return np.sqrt(x * x)


def logged(x: float) -> float:
    return np.log(max(x, 1e-9))


def compared(x: float) -> bool:
    return x > 0.5


def compared_guarded(x: float) -> bool:
    return not np.isnan(x) and x > 0.5


def test_sqrt_of_a_possibly_negative_value_is_info_and_safe_forms_are_quiet():
    assert fires(root, "nan_propagation").severity is Severity.INFO
    fires(compared, "nan_propagation")
    for fn in (root_of_square, logged, compared_guarded):
        quiet(fn, "nan_propagation")


# DCD109
def keep_going(bal: int) -> bool:
    return bal < 100


def keep_going_f32(bal: np.float32) -> bool:
    return bal < 100


@step(output="bal")
def compound(bal: int) -> int:
    return bal * bal


@step(output="bal")
def tick(bal: int) -> int:
    return bal + 1


@step(output="bal")
def narrow(bal: np.float32) -> np.float32:
    return bal + 1


def test_a_growing_int_carry_fires_but_a_counter_and_float64_do_not():
    assert len(checks.run(flow(loop(keep_going, compound, carries=["bal"], max_iterations=5, name="l"),
                               name="t"), ["accumulator_dtype"]).findings) == 1
    assert checks.run(flow(loop(keep_going, tick, carries=["bal"], max_iterations=5, name="l"),
                           name="t"), ["accumulator_dtype"]).findings == ()


def test_a_float32_carry_fires():
    report = checks.run(flow(loop(keep_going_f32, narrow, carries=["bal"], max_iterations=5, name="l"), name="t"),
                        ["accumulator_dtype"])
    assert [f.detail["dtype"] for f in report.findings] == ["float32"]


# DCD110
def widen(amount: Currency) -> float:
    return float(amount)


def narrowed(x: float) -> float:
    return np.int32(x)


def test_casts_that_lose_value_fire_and_plain_int_ops_are_quiet():
    fires(widen, "unsafe_cast")
    fires(narrowed, "unsafe_cast")
    quiet(sq, "unsafe_cast")


def test_numeric_checks_are_in_standard():
    assert {f"DCD1{i:02d}" for i in range(1, 11)} <= {c["code"] for c in checks.catalogue()}
    assert {"int64_overflow", "unsafe_cast"} <= set(checks.standard)
