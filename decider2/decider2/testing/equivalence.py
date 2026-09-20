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
from typing import Any

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
) -> None:
    meaning = _RUNG_MEANING[(left_name, right_name)]

    if left_exc is not None or right_exc is not None:
        if type(left_exc) is type(right_exc) and str(left_exc) == str(right_exc):
            return  # both modes agree by failing identically
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


def assert_equivalent(pipeline: Any, frame: pl.DataFrame, **kwargs: Any) -> None:
    """Run `pipeline.apply(frame, mode=..., **kwargs)` in all three modes
    (doc 02 §3.1) and assert EXACT agreement between them — promoting the
    ladder from something a handful of tests exercise incidentally to a
    first-class, reusable assertion any project can call.

    `**kwargs` is forwarded to `Pipeline.apply` verbatim (`params=`,
    `shared=`, `origin=`); do not pass `mode=` through it, since driving all
    three modes is this function's entire job.

    Raises `AssertionError` on the first disagreement (rung checked in the
    order `interpreted`/`stepped`, then `stepped`/`fused`), with a message
    that LOCALISES it to the rung it was introduced at (doc 02 §3.1's own
    taxonomy), naming the first disagreeing column and row and both values
    — or, if one mode raised where another didn't (or raised something
    different), naming that instead (see module docstring: this is a real
    outcome, not a hypothetical one — a boundary-value corpus's zero
    denominator triggers it directly).

    Returns `None` on success (three-way agreement).
    """
    if "mode" in kwargs:
        raise TypeError(
            "assert_equivalent() drives interpreted/stepped/fused itself; "
            "do not also pass mode= — that would run the same mode three "
            "times and defeat the point of the ladder."
        )

    interpreted_out, interpreted_exc = _run_mode(pipeline, frame, "interpreted", kwargs)
    stepped_out, stepped_exc = _run_mode(pipeline, frame, "stepped", kwargs)
    fused_out, fused_exc = _run_mode(pipeline, frame, "fused", kwargs)

    _assert_rung(
        "interpreted", interpreted_out, interpreted_exc,
        "stepped", stepped_out, stepped_exc,
    )
    _assert_rung(
        "stepped", stepped_out, stepped_exc,
        "fused", fused_out, fused_exc,
    )
