from __future__ import annotations

from itertools import repeat
from typing import Callable, Iterator

import numpy as np
import polars as pl

from decider.engine.boundary.nulls import MissingInputError
from decider.engine.ir.decls import Input, NullPolicy
from decider.engine.run.params import RunParams
from decider.engine.run.runners.base import Checkpoint
from decider.engine.run.state import State, dtype_of, from_series
from decider.engine.wiring.plan import Branch, Call, Loop, Plan, Resolved, Sequence, Version


class _Scope:
    """The rows a node runs on, the frame a frame step sees on them, and the names in sight."""

    __slots__ = ("rows", "base", "names")

    def __init__(self, rows: np.ndarray | None, base: pl.DataFrame, names: dict[str, Version]):
        self.rows = rows
        self.base = base
        self.names = names

    def count(self, n: int) -> int:
        return n if self.rows is None else len(self.rows)

    def child(self, keep: np.ndarray | None = None) -> _Scope:
        if keep is None:
            return _Scope(self.rows, self.base, dict(self.names))
        rows = np.flatnonzero(keep) if self.rows is None else self.rows[keep]
        base = self.base.filter(pl.Series(keep)) if self.base.width else self.base
        return _Scope(rows, base, dict(self.names))


class InterpretedRunner:
    """Runs every node in plain Python, row by row; row nodes call their Python `reference`.

    Args:
        visit: called with each locator a row node's `reference` reports
            (a tree's internal nodes); ignored by default.

    Example::

        for checkpoint in InterpretedRunner().iterate(plan, state, params):
            ...
    """

    def __init__(self, visit: Callable[[str], None] | None = None):
        self.visit = visit or _ignore

    def iterate(self, plan: Plan, state: State, params: RunParams) -> Iterator[Checkpoint]:
        root = _Scope(None, state.frame, {})
        yield from self._node(plan.root, state, params, root)
        state.frame = root.base

    def _node(self, r: Resolved, state: State, params: RunParams, scope: _Scope) -> Iterator[Checkpoint]:
        origin = r.node.origin
        yield Checkpoint(origin, "before")
        if isinstance(r, Call):
            self._call(r, state, params, scope)
        elif isinstance(r, Sequence):
            for child in r.children:
                yield from self._node(child, state, params, scope)
        elif isinstance(r, Branch):
            yield from self._branch(r, state, params, scope)
        else:
            yield from self._loop(r, state, params, scope)
        yield Checkpoint(origin, "after")

    def _call(self, call: Call, state: State, params: RunParams, scope: _Scope) -> None:
        node = call.node
        if node.kind == "frame":
            return _frame(call, state, scope)
        m = scope.count(state.n)
        bundle = params.bundle(call.id, m)
        cols = [_argument(state, v, i, scope.rows, node.origin.path) for i, v in zip(node.inputs, call.reads)]
        rows = zip(*cols) if cols else repeat((), m)
        if node.kind == "scalar":
            args = [i.arg for i in node.inputs]
            fixed = {**dict(node.consts), **{d.arg: b for d, b in zip(node.params, bundle)}}
            results = [node.fn(**dict(zip(args, row)), **fixed) for row in rows]
            if len(node.outputs) == 1:
                results = [(r,) for r in results]
        else:
            consts = tuple(v for _, v in node.consts)
            if node.reference is not None:
                results = [node.reference(row, bundle, consts, self.visit) for row in rows]
            else:
                results = [node.fn(row, bundle, consts) for row in rows]
        columns = list(zip(*results)) if results else [()] * len(node.outputs)
        if len(columns) != len(node.outputs):
            raise ValueError(f"{node.origin.path}: returned {len(columns)} values per row, "
                             f"but declares {len(node.outputs)} outputs")
        for v, out, values in zip(call.writes, node.outputs, columns):
            state.write(v, _array(values, dtype_of(out.annotation)), scope.rows)
            scope.names[v.name] = v

    def _branch(self, branch: Branch, state: State, params: RunParams, scope: _Scope) -> Iterator[Checkpoint]:
        cond = scope.child()
        yield from self._node(branch.condition, state, params, cond)
        picked, _ = state.read(branch.condition.writes[0], scope.rows)
        arm = np.where(picked, 0, 1) if picked.dtype == bool else picked.astype(np.int64)
        n_arms = len(branch.arms)
        bad = (arm < 0) | (arm >= n_arms)
        if bad.any():
            raise ValueError(f"branch {branch.node.origin.path}: the condition picked arm "
                             f"{int(arm[bad][0])} on {int(bad.sum())} row(s), but there are {n_arms} arms")
        taken = []
        for k, resolved in enumerate(branch.arms):
            keep = arm == k
            if not keep.any():
                continue
            inner = cond.child(keep)
            yield from self._node(resolved, state, params, inner)
            taken.append((k, inner.rows))
        for merge in branch.merges:
            for k, rows in taken:
                source = merge.arms[k] or merge.prior
                values, valid = state.read(source, rows)
                state.write(merge.version, values, rows, valid)
            scope.names[merge.version.name] = merge.version

    def _loop(self, loop: Loop, state: State, params: RunParams, scope: _Scope) -> Iterator[Checkpoint]:
        active = scope.child()
        for carry in loop.carries:
            values, valid = state.read(carry.initial, scope.rows)
            # A copy: later iterations write into the carried array in place.
            state.write(carry.version, values.copy(), scope.rows, valid)
            active.names[carry.version.name] = carry.version
        # ponytail: rows still looping at max_iterations stop silently; raise or flag them if that hides bugs.
        for _ in range(loop.node.max_iterations):
            cond = active.child()
            yield from self._node(loop.condition, state, params, cond)
            going, _ = state.read(loop.condition.writes[0], active.rows)
            going = going.astype(bool)
            if not going.any():
                break
            active = active.child(going)
            yield from self._node(loop.body, state, params, active)
            for carry in loop.carries:
                values, valid = state.read(carry.last, active.rows)
                state.write(carry.version, values, active.rows, valid)
        for carry in loop.carries:
            scope.names[carry.version.name] = carry.version


def _ignore(locator: str) -> None:
    pass


def _argument(state: State, version: Version, decl: Input, rows: np.ndarray | None, path: str) -> np.ndarray:
    values, valid = state.read(version, rows)
    if valid is None or valid.all():
        return values
    missing = ~valid
    if decl.null_policy is NullPolicy.REQUIRED:
        absent = version.producer is None and version.name not in state.frame.columns
        raise MissingInputError(decl.name, path, int(missing.sum()), len(valid), absent=absent)
    if decl.null_policy is NullPolicy.MISSING_AS:
        return np.where(valid, values, decl.fill)
    values = values.astype(object)
    values[missing] = None
    return values


def _array(values: tuple, dtype: np.dtype) -> np.ndarray:
    if dtype != object:
        return np.array(values, dtype)
    # Filled one by one so a tuple or list value stays one element.
    out = np.empty(len(values), object)
    for i, v in enumerate(values):
        out[i] = v
    return out


def _frame(call: Call, state: State, scope: _Scope) -> None:
    node = call.node
    path = node.origin.path
    df = state.frame_of(scope.base, scope.names, scope.rows)
    out = node.fn(df)
    if not isinstance(out, pl.DataFrame):
        raise TypeError(f"frame step {path} returned {type(out).__name__}, not a polars DataFrame")
    if out.height != df.height:
        raise ValueError(f"frame step {path} returned {out.height} rows for {df.height}; "
                         "frame steps must keep every row, in order")
    missing = [v.name for v in call.writes if v.name not in out.columns]
    if missing:
        raise ValueError(f"frame step {path} returned no column {missing[0]!r}, which "
                         f"{'it declares' if node.outputs is not None else 'later steps read'}; "
                         f"it returned {out.columns}")
    for v in call.writes:
        values, valid = from_series(out[v.name])
        state.write(v, values, scope.rows, valid)
    if node.outputs is None:
        # Unknown lineage: its frame is all that later nodes see.
        scope.base, scope.names = out, {}
    else:
        scope.names.update((v.name, v) for v in call.writes)
