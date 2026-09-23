from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import polars as pl
from polars.testing import assert_frame_equal

from decider.engine import Engine
from decider.engine.debug import Session

MODES = ("interpreted", "stepped", "fused")


def assert_equivalent(
    step: Any,
    frame: pl.DataFrame,
    params: Mapping[str, Any] | None = None,
    modes: tuple[str, ...] = MODES,
    script: Callable[[Session], Any] | None = None,
) -> pl.DataFrame:
    """Assert every mode gives exactly the same answer; returns the output frame.

    For each mode it checks, against the first mode's `run(frame)`:

    - `run(frame)`: same columns, dtypes and values (NaN equals NaN, nothing
      else is approximate);
    - `score(record)` for every row: the same dict as that row of `run`;
    - a debug `Session` resumed to the end: the same frame as `run`;
    - with `script`, a second session driven by `script(session)` (then
      resumed to the end): the same frame in every mode.

    An error in the first mode propagates as it is; `run` raising only in a
    later mode is a divergence like any other.

    Example::

        assert_equivalent(pipeline, df, params={"term": {"cap": 36.0}})

        def override(s):
            s.break_at("affordability_ratio")
            s.resume()
            s.set("disposable_income", 1200.0)

        assert_equivalent(pipeline, df, script=override)
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"assert_equivalent needs a polars DataFrame, not {type(frame).__name__}")
    expected = scripted = None
    for mode in modes:
        exe = Engine().bind(step, mode=mode)
        try:
            out = exe.run(frame, params)
        except Exception as e:
            if expected is None:
                raise
            raise AssertionError(f"run() raised {e!r} in {mode} mode but not in {modes[0]}") from e
        if expected is None:
            expected = out
        _same_frame(out, expected, f"run() in {mode} mode differs from {modes[0]}")
        for i, (record, row) in enumerate(zip(frame.iter_rows(named=True), out.iter_rows(named=True))):
            _same_record(exe.score(record, params), row, f"score() of row {i} in {mode} mode differs from run()")
        session = exe.session(frame, params)
        session.resume()
        _same_frame(_finish(session), out, f"a session resumed to the end in {mode} mode differs from run()")
        if script is not None:
            session = exe.session(frame, params)
            script(session)
            done = _finish(session)
            if scripted is None:
                scripted = done
            _same_frame(done, scripted, f"the scripted session in {mode} mode differs from {modes[0]}")
    return expected


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
            raise AssertionError(f"{what} at {key!r}: score={got[key]!r}, run={value!r}")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b
