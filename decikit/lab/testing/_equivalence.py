from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Mapping, Sequence

import polars as pl
from polars.testing import assert_frame_equal

from decikit import Engine, NumbaEngine
from decikit.debug import Session
from decikit.exceptions import EngineError
from decikit.ir import CallNode, iter_nodes, to_ir


def assert_equivalent(
    step: Any,
    frame: pl.DataFrame,
    params: Mapping[str, Any] | None = None,
    engines: Sequence[str | Engine] | None = None,
    script: Callable[[Session], Any] | None = None,
) -> pl.DataFrame:
    """Assert every engine gives exactly the same answer; returns the first engine's output frame.

    `engines` holds tags or `Engine` instances, by default every built-in
    engine: interpreted, stepped, fused and polars. Against the first
    engine's `run(frame)` each one checks:

    - `run(frame)` (on polars, `run(frame.lazy()).collect()`): same columns,
      dtypes and values (NaN equals NaN, nothing else is approximate);
    - `run(record)` for every row, as a dict: the same dict as that row of its
      `run(frame)`, or, when the pipeline has a frame step, as its run of a
      one-row frame of that record, since a cross-row step sees only that row;
    - on the numba engines, a debug `Session` resumed to the end: the same
      frame as `run`; with `script`, a second session driven by
      `script(session)` (then resumed to the end): the same frame on every
      numba engine.

    An error on the first engine propagates as it is; raising only on a later
    engine is a divergence. A later engine that refuses the pipeline with an
    `EngineError` (polars and an untyped `dict` output, say) is skipped with
    a warning naming it and the reason.

    Example::

        assert_equivalent(pipeline, df, params={"term": {"cap": 36.0}})
        assert_equivalent(pipeline, df, engines=("fused", "polars"))

        def override(s):
            s.break_at("affordability_ratio")
            s.resume()
            s.set("disposable_income", 1200.0)

        assert_equivalent(pipeline, df, script=override)
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"assert_equivalent needs a polars DataFrame, not {type(frame).__name__}")
    chosen = [Engine.resolve(e)() if isinstance(e, str) else e for e in engines or _built_ins()]
    first = chosen[0].type
    # A cross-row step answers a one-record frame differently from a many-record one, legitimately.
    one_row = any(isinstance(n, CallNode) and n.kind == "frame" for n in iter_nodes(to_ir(step)))
    expected = scripted = None
    for engine in chosen:
        name, numba = engine.type, isinstance(engine, NumbaEngine)
        try:
            exe = engine.bind(step)
            out = _run(exe, frame, params, numba)
        except EngineError as e:
            if expected is None:
                raise
            warnings.warn(f"assert_equivalent skipped the {name} engine, which refuses this pipeline: {e}", stacklevel=2)
            continue
        except Exception as e:
            if expected is None:
                raise
            raise AssertionError(f"run() raised {e!r} on the {name} engine but not on {first}") from e
        if expected is None:
            expected = out
        _same_frame(out, expected, f"run() on the {name} engine differs from {first}")
        rows = ((_run(exe, frame.slice(i, 1), params, numba).row(0, named=True) for i in range(frame.height))
                if one_row else out.iter_rows(named=True))
        for i, (record, row) in enumerate(zip(frame.iter_rows(named=True), rows)):
            _same_record(exe.run(record, params), row, f"run(record) of row {i} on the {name} engine differs from "
                         f"{'a one-row ' if one_row else ''}run(frame)")
        if not numba:
            continue
        session = exe.session(frame, params)
        session.resume()
        _same_frame(_finish(session), out, f"a session resumed to the end on the {name} engine differs from run()")
        if script is not None:
            session = exe.session(frame, params)
            script(session)
            done = _finish(session)
            if scripted is None:
                scripted = done
            _same_frame(done, scripted, f"the scripted session on the {name} engine differs from {first}")
    return expected


def _built_ins() -> list[str]:
    # Registration order puts interpreted first, the plainest reference.
    return [tag for tag, cls in Engine.registered().items() if cls.__module__.startswith("decikit.")]


def _run(exe: Any, frame: pl.DataFrame, params: Any, numba: bool) -> pl.DataFrame:
    return exe.run(frame, params) if numba else exe.run(frame.lazy(), params).collect()


def _finish(session: Session) -> pl.DataFrame:
    while not session.finished:
        session.resume()
    return session.output()


def _same_frame(got: pl.DataFrame, want: pl.DataFrame, what: str) -> None:
    try:
        assert_frame_equal(got, want, check_exact=True)
    except AssertionError as e:
        raise AssertionError(f"{what}: {e}") from None


def _same_record(got: dict, want: dict, what: str) -> None:
    if list(got) != list(want):
        raise AssertionError(f"{what}: keys {list(got)} != {list(want)}")
    for key, value in want.items():
        if not _same(got[key], value):
            raise AssertionError(f"{what} at {key!r}: record={got[key]!r}, frame={value!r}")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b
