"""The equivalence ladder as a first-class, reusable assertion — doc 02 §3.1,
doc 05 §9 criterion 3.

`interpreted ≡ stepped ≡ fused` is "the framework's core correctness
test" (doc 02 §3.1, verbatim), but before this module it had no home: it was
only exercised incidentally, once per test file, by hand-rolled comparisons
like `frames[0].equals(frames[1])` (`tests/test_runtime_invoke.py`) that say
THAT two frames disagree but not WHERE, let alone which rung introduced it.

Doc 05 §9.1 makes exact equality the acceptance criterion, not a tolerance —
measured, not assumed: EXPERIMENTS.md §I found 100.00% bit-exact agreement
over 20,000 rows with `fastmath` off, and traced the one real divergence
source (`fastmath`) to a *specific, excludable* opt-in, not to floating-point
computation in general. So a mismatch here means something is actually
wrong, and doc 02 §3.1 already names which layer to suspect from which
rung disagrees:

    interpreted != stepped -> numba changed a step's semantics
                               (float precision, integer overflow, division)
    stepped     != fused   -> fusion or inlining changed something

Without the middle rung, "compiled disagrees with Python" gives no way to
tell which of those it is — that is the entire reason `stepped` exists
(doc 02 §3.1: "structurally identical to the fallback path", not a separate
thing to maintain).

Division is not a hypothetical example of "numba changed a step's
semantics": a boundary-value frame (`decider2.testing.corpus`) routinely
produces a zero denominator, and CPython float division and numba's default
(Python-compatible) error model both raise `ZeroDivisionError` on `0.0/0.0`
— but `interpreted` mode reads its row values out of a numpy array
(`decider2.runtime.modes._row_kwargs`), so the same `0.0/0.0` runs as a
*numpy scalar* division, which is silent IEEE-754 (`nan`, no exception).
Concretely: on the SAME boundary row, `interpreted` can return a value while
`stepped`/`fused` raise. This module treats "one mode raised, another
didn't" (or raised something different) as a divergence like any other, not
an unhandled crash — it is exactly the kind of thing this ladder exists to
localise.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import polars as pl

_MODES: tuple[str, ...] = ("interpreted", "stepped", "fused")

_RUNG_MEANING = {
    ("interpreted", "stepped"): (
        "numba changed a step's semantics — float precision, integer "
        "overflow, or division"
    ),
    ("stepped", "fused"): "fusion or inlining changed something",
}


def _run_mode(
    pipeline: Any, frame: pl.DataFrame, mode: str, kwargs: dict
) -> tuple[pl.DataFrame | None, Exception | None]:
    """`(output, None)` on success, `(None, exc)` on failure — a raised
    exception is comparable data here, not something to let escape and
    crash the ladder before it can say which rung it happened at."""
    try:
        return pipeline.apply(frame, mode=mode, **kwargs), None
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see module docstring
        return None, exc


def _values_equal(a: Any, b: Any) -> bool:
    """Exact equality, except NaN == NaN.

    Doc 05 §9.1's "exact equality" means three runs of the SAME computation
    agree bit-for-bit — and if a row's honest answer is NaN (e.g. a routed
    division), all three modes producing NaN at that row IS agreement, not a
    divergence to report. Plain `==` would call that a mismatch (IEEE-754:
    `nan != nan`), which is the wrong thing for this specific job.
    """
    try:
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
    except TypeError:
        pass
    return a == b


def _first_divergence(left: pl.DataFrame, right: pl.DataFrame) -> tuple[str, int, Any, Any] | None:
    """The first `(column, row, left_value, right_value)` at which `left`
    and `right` disagree, walking columns in `left`'s order and rows in
    order within each column — or `None` if every shared column agrees.

    Only columns present in both are compared. `assert_equivalent` always
    calls this on two successful outputs of the SAME pipeline/frame/params,
    which by construction produce the same column set (doc 03 §7's additive
    frame doesn't depend on `mode`) — restricting to the intersection just
    means a caller reusing this helper elsewhere doesn't get a spurious
    KeyError over an incidental column difference instead of the real
    answer.
    """
    common = [c for c in left.columns if c in right.columns]
    for col in common:
        lv = left[col].to_list()
        rv = right[col].to_list()
        for i, (a, b) in enumerate(zip(lv, rv)):
            if not _values_equal(a, b):
                return col, i, a, b
    return None


def _describe_row(frame: pl.DataFrame, row: int) -> str:
    """`decider2.testing.corpus` tags every row it generates with a `case`
    column naming the boundary value it exercises (`"zero:net_income"`,
    `"null_edge:expenses"`, ...). When the frame under test carries one,
    naming it turns "row 17 disagrees" into "row 17 (which was
    'zero:net_income') disagrees" — the caller doesn't have to go re-index
    the frame by hand to find out what was actually being tested."""
    if "case" in frame.columns:
        return f"row {row} (case={frame['case'][row]!r})"
    return f"row {row}"


def _assert_rung(
    left_name: str, left_out: pl.DataFrame | None, left_exc: Exception | None,
    right_name: str, right_out: pl.DataFrame | None, right_exc: Exception | None,
) -> Exception | None:
    """Checks one adjacent pair of rungs. Raises `AssertionError` on a
    genuine divergence (unchanged). Returns `None` when this pair produced
    real, matching output — true agreement. Returns the shared exception
    when this pair "agreed" only by crashing identically (review finding
    3): that is NOT yet a verdict — `assert_equivalent` below still has to
    check whether the OTHER adjacent pair also crashed, because only when
    every rung crashes identically is there truly nothing to compare, and
    even that case must fail loudly rather than pass silently (see
    `assert_equivalent`'s docstring).
    """
    meaning = _RUNG_MEANING[(left_name, right_name)]

    if left_exc is not None or right_exc is not None:
        if type(left_exc) is type(right_exc) and str(left_exc) == str(right_exc):
            return left_exc  # this PAIR agrees, but only by both crashing
        raise AssertionError(
            f"equivalence ladder broke between {left_name} and {right_name} "
            f"(doc 02 §3.1: {meaning}): "
            f"{left_name}={'raised ' + repr(left_exc) if left_exc else 'succeeded'}, "
            f"{right_name}={'raised ' + repr(right_exc) if right_exc else 'succeeded'}."
        )

    divergence = _first_divergence(left_out, right_out)
    if divergence is not None:
        col, row, left_value, right_value = divergence
        raise AssertionError(
            f"equivalence ladder broke between {left_name} and {right_name} "
            f"(doc 02 §3.1: {meaning}) at column {col!r}, "
            f"{_describe_row(left_out, row)}: "
            f"{left_name}={left_value!r}, {right_name}={right_value!r}."
        )
    return None


def _assert_score_agrees_with_apply(
    pipeline: Any, frame: pl.DataFrame, reference: pl.DataFrame, kwargs: Mapping[str, Any]
) -> None:
    """The fourth rung (review finding 3; doc 05 §9 criterion 2: "the same
    kernel answers a single record"). Nothing checked this before —
    `runtime.invoke.score`'s own module docstring calls its extraction a
    "minimal reference implementation" with no `decider2.boundary`
    equivalent to call, so agreement with the batch path is not free the
    way interpreted/stepped/fused's *shared* kernel makes THEIR agreement
    close to free.

    Runs only after the three-mode ladder above has already established
    `reference` (the fused-mode frame) as a value every mode agrees on —
    safe ground truth for a single row. Only `params=`/`shared=` forward to
    `score()`; `Pipeline.score` takes no `mode=`/`origin=`.

    A row `score()` routes (doc 03 §1 — an absent/null REQUIRED input) is
    skipped rather than compared: `score()`'s per-record extraction and
    `apply()`'s whole-frame extraction are independent implementations that
    can route on different grounds (see the module docstrings), and
    rendering a routed `Decision` against `apply()`'s placeholder terminal
    value (`_scatter_back`'s NaN/0/False) is `observe/`'s job, not yet
    built (doc 00-BUILD.md Layer 5) — a different check than this rung.

    A pipeline with any `str`-typed input is skipped entirely, not
    compared: `runtime.invoke.score`'s own module docstring calls its
    extraction a "minimal reference implementation" with "no dtype ladder"
    at all, so a `str` input is out of its scope from the start — a
    pre-existing, separately-scoped gap (doc 05 §1.5's dictionary-code
    encoding has no single-record equivalent) that review finding 3's fix
    is not asking to close.
    """
    interface = getattr(pipeline, "interface", None)
    if interface is not None and any(inp.annotation is str for inp in interface.inputs):
        return
    score_kwargs = {k: v for k, v in kwargs.items() if k in ("params", "shared")}
    terminal_names = [c for c in reference.columns if c not in frame.columns]
    if not terminal_names:
        return
    for i, record in enumerate(frame.to_dicts()):
        result = pipeline.score(record, **score_kwargs)
        if "decision" in result and not any(name in result for name in terminal_names):
            continue  # this row routed under score(); not this rung's question
        for name in terminal_names:
            if name not in result:
                continue
            expected = reference[name][i]
            actual = result[name]
            if not _values_equal(expected, actual):
                raise AssertionError(
                    "equivalence ladder broke between fused (batch) and "
                    "score (doc 05 §9 criterion 2: 'the same kernel answers "
                    f"a single record') at column {name!r}, "
                    f"{_describe_row(frame, i)}: apply={expected!r}, "
                    f"score={actual!r}."
                )


def assert_equivalent(pipeline: Any, frame: pl.DataFrame, **kwargs: Any) -> None:
    """Run `pipeline.apply(frame, mode=..., **kwargs)` in all three modes
    (doc 02 §3.1), then `pipeline.score()` per row, and assert EXACT
    agreement across all four — promoting the ladder from something a
    handful of tests exercise incidentally to a first-class, reusable
    assertion any project can call.

    `**kwargs` is forwarded to `Pipeline.apply` verbatim (`params=`,
    `shared=`, `origin=`); do not pass `mode=` through it, since driving all
    three modes is this function's entire job. `frame` must be a real
    `polars.DataFrame` (review finding 3): passing anything else — a dict,
    say — used to raise the SAME exception in every mode, which this
    function's own bug then counted as three-way agreement despite never
    having run the pipeline at all.

    Raises `AssertionError` on the first disagreement (rung checked in the
    order `interpreted`/`stepped`, then `stepped`/`fused`, then `fused`/
    `score`), with a message that LOCALISES it to the rung it was
    introduced at (doc 02 §3.1's own taxonomy), naming the first
    disagreeing column and row and both values — or, if one mode raised
    where another didn't (or raised something different), naming that
    instead (see module docstring: this is a real outcome, not a
    hypothetical one — a boundary-value corpus's zero denominator triggers
    it directly).

    Review finding 3's other bug: if EVERY mode raises the identical
    exception, that is not agreement either — it is a total failure with
    nothing to compare (three identical `AttributeError`s, say, from a
    pipeline that never ran at all). This function used to return `None`
    for that case; it now re-raises the shared exception instead, so a
    caller sees the real failure rather than a false green.

    Returns `None` on success (four-way agreement).
    """
    if "mode" in kwargs:
        raise TypeError(
            "assert_equivalent() drives interpreted/stepped/fused itself; "
            "do not also pass mode= — that would run the same mode three "
            "times and defeat the point of the ladder."
        )
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(
            "assert_equivalent(pipeline, frame, ...) needs a polars "
            f"DataFrame for `frame`, not {type(frame).__name__!r} (review "
            "finding 3: a non-DataFrame used to make every mode raise the "
            "same exception, which this function's own bug then counted as "
            "agreement instead of the vacuous run it was)."
        )

    interpreted_out, interpreted_exc = _run_mode(pipeline, frame, "interpreted", kwargs)
    stepped_out, stepped_exc = _run_mode(pipeline, frame, "stepped", kwargs)
    fused_out, fused_exc = _run_mode(pipeline, frame, "fused", kwargs)

    crash1 = _assert_rung(
        "interpreted", interpreted_out, interpreted_exc,
        "stepped", stepped_out, stepped_exc,
    )
    crash2 = _assert_rung(
        "stepped", stepped_out, stepped_exc,
        "fused", fused_out, fused_exc,
    )
    if crash1 is not None or crash2 is not None:
        # Every rung "agreed" only by raising the identical exception --
        # interpreted, stepped and fused all crashed the same way. A
        # divergence between just one adjacent pair already raised an
        # AssertionError above before reaching here, so by the time BOTH
        # `_assert_rung` calls return without raising, either both are None
        # (genuine three-way value agreement, handled below) or both are
        # set (all three crashed identically) -- never a mix.
        raise crash1 if crash1 is not None else crash2  # type: ignore[misc]

    _assert_score_agrees_with_apply(pipeline, frame, fused_out, kwargs)
