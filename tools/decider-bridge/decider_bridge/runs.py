"""Whole runs for comparisons, and per-record views into one node of a live session."""
from __future__ import annotations

import math

import polars as pl

from decider.engine import Engine

from .controls import Controls


def apply_overrides(frame: pl.DataFrame, overrides: dict | None, row: int | None) -> pl.DataFrame:
    """`frame` with each column in `overrides` set to its value, on `row` or on every row."""
    for name, value in (overrides or {}).items():
        if row is None:
            frame = frame.with_columns(pl.lit(value).alias(name))
        else:
            values = frame[name].to_list() if name in frame.columns else [None] * frame.height
            values[row] = value
            frame = frame.with_columns(pl.Series(name, values, strict=False))
    return frame


def trace(step, frame: pl.DataFrame, params: dict | None, ir: dict | None = None, forces=()) -> dict:
    """Run to the end and keep what every call wrote, for a step-by-step diff against another run.

    `forces` (branch arms or loop iteration counts, see `Controls.set`) need the described `ir`.
    """
    session = Engine().bind(step).session(frame, params)
    if forces:
        steer(session, ir, forces)
    error = None
    try:
        session.resume()
        while not session.finished:
            session.resume()
    except Exception as e:  # the trace up to the failing step is still worth comparing
        error = f"{type(e).__name__}: {e}"
    return collect(session, error)


def steer(session, ir, forces):
    c = Controls(ir)
    c.set(forces)
    return c.attach(session)


def collect(session, error):
    """What every call of a run wrote, and the output if it finished."""
    state = session.state
    steps = {}
    for call in session.executable.plan.calls:
        written = {v.name: state.column(v.name, v).to_list() for v in call.writes if v.id in state.values}
        if written:
            steps[call.node.origin.path] = written
    output = None if error else {c: s.to_list() for c, s in session.output().to_dict().items()}
    return {"steps": steps, "output": output, "error": error}


def _call(session, path):
    for call in session.executable.plan.calls:
        if call.node.origin.path == path:
            return call
    raise KeyError(f"no call node at {path!r}")


def _row_inputs(session, call, row):
    """The values `call` reads on `row`, nulls filled the way the runner fills them."""
    out = []
    for decl, v in zip(call.node.inputs, call.reads):
        values, valid = session.state.read(v)
        value = None if valid is not None and not valid[row] else values[row]
        if value is None and decl.fill is not None:
            value = decl.fill
        out.append(value.item() if hasattr(value, "item") else value)
    return out


def tree_path(session, path: str, row: int) -> dict:
    """Re-walk a row node's Python reference for one record: the positions it reaches, in order.

    Steps are pure, so walking again gives the path the run took.
    """
    call = _call(session, path)
    node = call.node
    if node.reference is None:
        raise ValueError(f"{path} has no Python reference to walk")
    visited: list[str] = []
    # ponytail: reads the session's private params; ask Session for a public bundle accessor if this sticks.
    bundle = session._params.bundle(call.id, 1)
    result = node.reference(tuple(_row_inputs(session, call, row)), bundle, tuple(v for _, v in node.consts), visited.append)
    return {"path": path, "row": row, "visited": visited, "result": list(result)}


def debug_condition(session, path: str, row: int) -> str | None:
    """A debugpy breakpoint condition that holds only when the step is called with `row`'s inputs."""
    call = _call(session, path)
    node = call.node
    if node.inputs is None:
        return None
    values = _row_inputs(session, call, row)
    if node.kind == "row":
        return f"row == {tuple(values)!r}"
    tests = []
    for decl, value in zip(node.inputs, values):
        if isinstance(value, float) and math.isnan(value):
            continue  # nan never equals itself
        tests.append(f"{decl.arg} is None" if value is None else f"{decl.arg} == {value!r}")
    # ponytail: rows with identical inputs all match; that is the same computation, so it rarely matters.
    return " and ".join(tests) or None
