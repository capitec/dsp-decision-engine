"""New tests for string support — doc 05 §1.5 ("Strings in detail") and
EXPERIMENTS.md §O.

Not part of the protected `test_spec_conformance.py`/`test_flagship.py`
suites. `test_spec_conformance.py` already exercises the end-to-end shape
(`sector_rate`, all three modes, a params-only literal change, the absent-
literal sentinel). What is missing there — and what this file pins — is the
one property EXPERIMENTS.md §O and doc 08 §2.1 say the whole design rests
on: a string literal is a kernel *argument* (an int32 dictionary code), so
changing which literal a `str` param resolves to is a value change, never a
recompile. `test_compile_driver.py::test_retuning_never_grows_driver_
signatures` pins the numeric version of this claim; this file pins the
string version the same way, at the same layer (`build_driver` + `modes.
run_fused`, module-level step functions — a closure-defined step can't be
imported by the generated kernel file, doc 05 §4.1, and would silently land
as a `fallback` segment instead of proving anything about `kernel.
signatures`).

The remaining tests exercise `runtime.invoke.resolve_params`'s new str-param
encoding directly (categories -> code, absent -> sentinel, ambiguous column,
no str input at all) — the unit-level counterparts of the error/sentinel
behaviour `test_spec_conformance.py` only checks end to end.
"""
from __future__ import annotations

import numpy as np
import pytest

from decider2.compile.driver import build_driver
from decider2.runtime import invoke, modes
from decider2.types import Input, ParamDecl, Step


def sector_rate(sector: str, private: str, rate: float) -> float:
    """Module-level on purpose — see module docstring."""
    return rate if sector == private else 1.0


def _sector_rate_step() -> Step:
    return Step(
        name="sector_rate",
        fn=sector_rate,
        inputs=(Input("sector", str),),
        params=(
            ParamDecl("private", str, "private", None),
            ParamDecl("rate", float, 0.9, None),
        ),
    )


# ---------------------------------------------------------------------------
# The property the whole design rests on
# ---------------------------------------------------------------------------


def test_a_string_literal_param_never_grows_kernel_signatures(tmp_path):
    """EXPERIMENTS.md §O, verbatim: 'len(kernel.signatures) stays at 1
    across three distinct literal sets.' Doc 08 §2.1's guarantee, for
    strings: retuning `private` from one real code to another, and then to
    the sentinel for a literal absent from the data, must all land on the
    SAME numba specialisation — the literal never becomes an emitted source
    constant, only ever a kernel argument.
    """
    step = _sector_rate_step()
    driver = build_driver(
        [step], [0], build_dir=tmp_path, terminal_names=frozenset({"sector_rate"})
    )
    assert driver.segments[0].kind == "compiled"  # must actually compile to mean anything

    n = 4
    sector_codes = np.array([0, 1, 0, 2], dtype=np.int32)  # private, public, private, government

    for private_code in (np.int32(0), np.int32(2), np.int32(-1)):
        resolved = modes.ResolvedParams(
            per_step_scalar={
                ("sector_rate", "private"): private_code,
                ("sector_rate", "rate"): 0.9,
            },
            per_step_bundle={},
        )
        out = modes.run_fused(driver, {"sector": sector_codes}, resolved, n)
        assert out["sector_rate"].dtype == np.float64  # sanity: it actually ran

    assert len(driver.signatures) == 1


# ---------------------------------------------------------------------------
# resolve_params: str-param encoding, unit level
# ---------------------------------------------------------------------------

_CATEGORIES = {"sector": ("private", "public", "government")}


def test_str_param_resolves_through_the_columns_categories():
    step = _sector_rate_step()
    resolved = invoke.resolve_params(
        [step],
        {"sector_rate": {"private": "government", "rate": 0.9}},
        categories=_CATEGORIES,
    )
    code = resolved.per_step_scalar[("sector_rate", "private")]
    assert code == np.int32(2)
    assert isinstance(code, np.int32)


def test_a_literal_absent_from_categories_resolves_to_the_sentinel_not_an_error():
    step = _sector_rate_step()
    resolved = invoke.resolve_params(
        [step],
        {"sector_rate": {"private": "martian", "rate": 0.9}},
        categories=_CATEGORIES,
    )
    assert resolved.per_step_scalar[("sector_rate", "private")] == np.int32(-1)


def test_a_step_reading_two_str_inputs_is_a_clear_error_not_a_guess():
    """§O + task spec: 'the step reads several -> raise a clear error...;
    do not guess.'"""

    def two_sided(a: str, b: str, which: str = "a", rate: float = 1.0) -> float:
        return rate

    step = Step(
        name="two_sided",
        fn=two_sided,
        inputs=(Input("a", str), Input("b", str)),
        params=(
            ParamDecl("which", str, "a", None),
            ParamDecl("rate", float, 1.0, None),
        ),
    )
    with pytest.raises(ValueError, match="two_sided.*str-typed inputs"):
        invoke.resolve_params(
            [step],
            {"two_sided": {"which": "a", "rate": 1.0}},
            categories={"a": ("a", "b"), "b": ("a", "b")},
        )


def test_a_str_input_with_no_str_param_is_a_named_loud_error():
    """The negative control for the shape `test_spec_conformance.py::
    test_a_string_input_is_never_silently_zeroed` pins end to end: a step
    with a str input and no str param has no way to compare it against
    anything but a bare (unusable) literal. Checked unconditionally, with
    no params override supplied at all."""

    def bare_literal(sector: str) -> float:
        return 1.0 if sector == "private" else 0.0

    step = Step(name="bare_literal", fn=bare_literal, inputs=(Input("sector", str),), params=())
    with pytest.raises(ValueError, match="sector"):
        invoke.resolve_params([step], {}, categories=_CATEGORIES)
