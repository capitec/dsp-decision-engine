"""Tests for `decider2.testing` — doc 02 §3.1's equivalence ladder and doc 05
§9's acceptance criteria, built as first-class, reusable assertions (this
agent's Job 1) rather than something exercised only incidentally by a few
other test files (as `tests/test_runtime_invoke.py`'s
`test_the_three_modes_agree_exactly` and
`tests/test_runtime_modes.py`'s `test_the_three_modes_agree_exactly_when_
fused_into_one_kernel` already did, by hand, once each).

`test_flaky_global_step_breaks_the_ladder_and_is_localised_to_the_right_rung`
below is the NEGATIVE control the task asked for: a ladder that cannot fail
is not a test. It deliberately makes a step disagree across modes (reading a
mutable module global, which numba freezes at compile time — doc 02 §3.1's
own example of "numba changed a step's semantics") and asserts
`assert_equivalent` both catches it AND names the correct rung
(`interpreted != stepped`, not `stepped != fused`).
"""
from __future__ import annotations

import pytest
import polars as pl

from decider2 import flow, missing_as, not_applicable_as, param
from decider2.compile.driver import clear_driver_cache
from decider2.testing import assert_equivalent, assert_no_recompile, corpus

# --- a well-behaved, multi-null-tier fixture pipeline -----------------------
#
# Deliberately free of anything numba and CPython could disagree on (no
# division, no integer arithmetic that could overflow) so that running it
# over `corpus()`'s boundary values is a genuine positive case: all four
# null-policy tiers (doc 03 §1) are represented, plus an `int`-typed input,
# so `corpus()` has something of every kind to generate boundary rows for.


def net(a: float, b: float) -> float:
    """REQUIRED tier (a, b)."""
    return a - b


def scaled(net: float, factor: float = param(2.0, ge=0)) -> float:
    return net * factor


def big(n: int | None) -> bool:
    """OPTIONAL tier, and the only int-typed leaf (int64-near-2**53 case)."""
    return False if n is None else n > 1000


def padded(extra: float = missing_as(10.0)) -> float:
    """MISSING_AS tier."""
    return extra + 1.0


def bonus(spouse_income: float = not_applicable_as(0.0)) -> float:
    """NOT_APPLICABLE_AS tier."""
    return spouse_income * 0.5


def flagged(active: bool) -> float:
    """REQUIRED tier, bool-typed -- `corpus()` should emit a `zero`/
    `null_edge` row for this but no `negative` (bool has no meaningful
    negative)."""
    return 1.0 if active else 0.0


def _fixture_pipeline():
    return flow(net, scaled, big, padded, bonus, flagged)


FRAME = pl.DataFrame({
    "a": [10.0, 20.0, 5.0],
    "b": [3.0, 1.0, 5.0],
    "n": [500, 2000, None],
    "extra": [1.0, None, 3.0],
    "spouse_income": [None, 100.0, 0.0],
    "active": [True, False, True],
})


# --- assert_equivalent: the positive cases ----------------------------------


def test_assert_equivalent_passes_when_all_three_modes_agree():
    # No exception == success; this is the ordinary, unremarkable case doc
    # 05 §9.1 says should be exact, not approximate, agreement.
    assert assert_equivalent(_fixture_pipeline(), FRAME) is None


def test_assert_equivalent_passes_over_the_generated_boundary_corpus():
    pipeline = _fixture_pipeline()
    generated = corpus(pipeline)
    assert_equivalent(pipeline, generated["boundary"])


def test_assert_equivalent_passes_over_the_empty_corpus_frame():
    pipeline = _fixture_pipeline()
    generated = corpus(pipeline)
    assert generated["empty"].height == 0
    assert_equivalent(pipeline, generated["empty"])


def test_assert_equivalent_rejects_a_mode_kwarg():
    with pytest.raises(TypeError, match="mode"):
        assert_equivalent(_fixture_pipeline(), FRAME, mode="fused")


# --- the negative control: a step reading a mutable global ------------------
#
# Doc 02 §3.1: "interpreted != stepped -> numba changed a step's semantics."
# A step closing over a plain module-level global is the cleanest way to
# force exactly that: numba's njit specialises the function once (freezing
# whatever the global's value was AT COMPILE TIME, doc 00-BUILD.md's
# staleness family of gotchas) and never re-reads it, while `interpreted`
# mode calls the same Python function fresh on every row and sees whatever
# the global is NOW. Warming the driver (a `mode="fused"` call, which
# compiles every step regardless of which mode is eventually run) BEFORE
# mutating the global is what makes this deterministic rather than a race:
# `decider2.compile.driver.build_driver` explicitly calls `fn.compile(sig)`
# during the build (not lazily on first real call), so the freeze happens at
# that warm-up call, not at whatever moment a test happens to first invoke
# the compiled kernel.

_THRESHOLD = 5.0


def flaky(x: float) -> float:
    return x + _THRESHOLD


def test_flaky_global_step_breaks_the_ladder_and_is_localised_to_the_right_rung():
    global _THRESHOLD
    pipeline = flow(flaky)
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0]})

    pipeline.apply(frame, mode="fused")  # warm: freezes _THRESHOLD == 5.0
    _THRESHOLD = 999.0  # mutate after compile -- numba never sees this
    try:
        with pytest.raises(AssertionError) as excinfo:
            assert_equivalent(pipeline, frame)

        message = str(excinfo.value)
        # LOCALISED: names interpreted/stepped, not stepped/fused --
        # fusion/inlining had nothing to do with this divergence.
        assert "interpreted" in message and "stepped" in message
        assert "stepped and fused" not in message
        # Both concrete values are named, not just "they differ".
        assert "1000.0" in message  # interpreted: 1.0 + 999.0 (live global)
        assert "6.0" in message     # stepped/fused: 1.0 + 5.0 (frozen global)
    finally:
        _THRESHOLD = 5.0  # leave the module global as later runs expect it


# --- corpus() ----------------------------------------------------------------


def test_corpus_returns_a_boundary_frame_and_an_empty_frame():
    generated = corpus(_fixture_pipeline())
    assert set(generated) == {"boundary", "empty"}
    assert generated["boundary"].height > 0
    assert generated["empty"].height == 0
    assert generated["empty"].schema == generated["boundary"].schema


def test_corpus_covers_every_declared_input():
    pipeline = _fixture_pipeline()
    generated = corpus(pipeline)
    boundary = generated["boundary"]
    for inp in pipeline.interface.inputs:
        assert inp.name in boundary.columns
        cases = boundary["case"]
        assert any(c.startswith(f"zero:{inp.name}") for c in cases)
        assert any(c.startswith(f"null_edge:{inp.name}") for c in cases)


def test_corpus_zero_and_negative_rows_hit_the_named_column():
    boundary = corpus(_fixture_pipeline())["boundary"]
    zero_a = boundary.filter(pl.col("case") == "zero:a")
    assert zero_a["a"].to_list() == [0.0]

    negative_a = boundary.filter(pl.col("case") == "negative:a")
    assert negative_a["a"].item() < 0


def test_corpus_skips_negative_for_a_bool_input_but_not_for_an_int_one():
    boundary = corpus(_fixture_pipeline())["boundary"]
    cases = boundary["case"].to_list()
    # bool: no meaningful "negative" -- must not be fabricated.
    assert not any(c.startswith("negative:active") for c in cases)
    assert any(c.startswith("zero:active") for c in cases)
    assert any(c.startswith("null_edge:active") for c in cases)
    # int: `n` DOES get a negative case.
    assert any(c.startswith("negative:n") for c in cases)


def test_corpus_bool_zero_case_is_false():
    boundary = corpus(_fixture_pipeline())["boundary"]
    row = boundary.filter(pl.col("case") == "zero:active")
    assert row.height == 1
    assert row["active"].item() is False


def test_corpus_int_typed_input_gets_a_near_2_53_row():
    boundary = corpus(_fixture_pipeline())["boundary"]
    row = boundary.filter(pl.col("case") == "int64_near_2**53:n")
    assert row.height == 1
    assert row["n"].item() == 2**53 + 1


def test_corpus_float_typed_input_has_no_near_2_53_row():
    boundary = corpus(_fixture_pipeline())["boundary"]
    cases = boundary["case"].to_list()
    assert not any(c.startswith("int64_near_2**53:a") for c in cases)


def test_corpus_null_edge_is_a_genuine_null():
    boundary = corpus(_fixture_pipeline())["boundary"]
    row = boundary.filter(pl.col("case") == "null_edge:extra (missing_as)")
    assert row.height == 1
    assert row["extra"].item() is None


def test_corpus_rejects_a_source_with_no_declared_inputs():
    from decider2.types import Interface

    empty_interface = Interface(inputs=(), outputs=(), terminals=(), params_model=None)
    with pytest.raises(ValueError, match="no inputs"):
        corpus(empty_interface)


# --- assert_no_recompile -----------------------------------------------------


def _pricing_pipeline():
    return flow(net, scaled)


PRICING_FRAME = pl.DataFrame({"a": [10.0, 20.0], "b": [1.0, 2.0]})


def test_assert_no_recompile_passes_for_a_value_only_retune():
    assert assert_no_recompile(
        _pricing_pipeline(), PRICING_FRAME,
        {"scaled": {"factor": 2.0}}, {"scaled": {"factor": 3.0}},
    ) is None


class _RecompileInjectingPipeline:
    """Wraps a real `Pipeline`, forcing `decider2.compile.driver`'s in-
    process driver cache to be dropped between the two `.apply()` calls
    `assert_no_recompile` makes -- simulating a genuine (if artificial)
    recompile between a value-only retune, purely so the test below can
    prove `assert_no_recompile` actually has teeth (doc 05 §9's own
    "changing a field's type adds one" is a build-time negative control this
    function's signature can't express directly, since `params_a`/
    `params_b` are runtime values against ONE fixed, already-compiled
    pipeline -- this exercises the same underlying invariant the other way
    around: a driver rebuild between two `.apply()` calls IS what this
    function exists to catch, however it happens to be triggered).
    """

    def __init__(self, inner):
        self._inner = inner
        self._calls = 0

    def flatten_for_runtime(self):
        return self._inner.flatten_for_runtime()

    @property
    def interface(self):
        return self._inner.interface

    def apply(self, frame, **kwargs):
        self._calls += 1
        if self._calls == 2:
            clear_driver_cache()
        return self._inner.apply(frame, **kwargs)


def test_assert_no_recompile_catches_a_genuine_rebuild_between_calls():
    wrapped = _RecompileInjectingPipeline(_pricing_pipeline())
    with pytest.raises(AssertionError, match="rebuilt the driver"):
        assert_no_recompile(
            wrapped, PRICING_FRAME,
            {"scaled": {"factor": 2.0}}, {"scaled": {"factor": 3.0}},
        )
