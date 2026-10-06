"""Param checks: missing bounds, defaults outside them, params the body ignores."""
from typing import Annotated

from decider import flow, param
from decider.lab import checks
from decider.ops import Percent
from decider.lab.checks import Severity


def _found(fn, alias):
    return checks.run(flow(fn, name="f"), [alias]).findings


def test_an_unbounded_numeric_param_is_info():
    def rule(x: float, cap: float = param(5.0)) -> float:
        return min(x, cap)

    (f,) = _found(rule, "param_unconstrained")
    assert (f.code, f.severity, f.detail["param"]) == ("DCD301", Severity.INFO, "cap")


def test_a_percent_param_needs_both_bounds_of_a_fraction():
    def rule(x: float, a: Annotated[float, Percent()] = param(0.5, ge=0), b: Annotated[float, Percent()] = param(0.5, ge=0, le=1)) -> float:
        return x * a * b

    (f,) = _found(rule, "param_unconstrained")
    assert f.detail["param"] == "a" and f.severity is Severity.WARNING


def test_bounded_and_non_numeric_params_are_quiet():
    def rule(x: float, cap: float = param(5.0, ge=0), on: bool = param(True), name: str = param("a")) -> float:
        return min(x, cap) if on and name else 0.0

    assert not _found(rule, "param_unconstrained")


def test_a_default_outside_its_bounds_is_reported():
    def rule(x: float, cap: float = param(5.0, le=3)) -> float:
        return min(x, cap)

    (f,) = _found(rule, "param_default_out_of_bounds")
    assert (f.code, f.detail["kind"], f.detail["param"]) == ("DCD302", "default", "cap")


def test_silent_on_invalid_default_is_reported():
    def rule(x: float, cap: float = param(2.0, le=3, on_invalid="default")) -> float:
        return min(x, cap)

    (f,) = _found(rule, "param_default_out_of_bounds")
    assert f.detail["kind"] == "on_invalid"


def test_in_bounds_default_with_warn_is_quiet():
    def rule(x: float, cap: float = param(2.0, le=3, on_invalid="warn")) -> float:
        return min(x, cap)

    assert not _found(rule, "param_default_out_of_bounds")


def test_a_param_the_body_never_reads_is_reported():
    def rule(x: float, cap: float = param(5.0), floor: float = param(1.0)) -> float:
        return max(x, floor)

    (f,) = _found(rule, "param_unused")
    assert (f.code, f.detail["param"]) == ("DCD303", "cap")


def test_params_that_are_read_are_quiet():
    def rule(x: float, cap: float = param(5.0)) -> float:
        return min(x, cap)

    assert not _found(rule, "param_unused")
