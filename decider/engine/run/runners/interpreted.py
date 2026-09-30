from __future__ import annotations

import copy
from itertools import repeat
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import polars as pl

from decider.engine.boundary.nulls import MissingInputError
from decider.engine.compile import numpy_dtype
from decider.engine.compile.rows import build_rows, rows_needs_no_fill
from decider.engine.ir.decls import Input, NullPolicy, base_annotation
from decider.engine.run.params import RunParams
from decider.engine.run.representations import codes, span_objects
from decider.engine.run.runners.base import Checkpoint
from decider.engine.run.state import State, dtype_of, fill_missing, from_series, record_value
from decider.engine.trace.envelope import Kind
from decider.types import Representation, is_raw, representation_for, columnar_item, item_schema, struct_item
from decider.engine.wiring.plan import Branch, Call, Loop, Plan, Resolved, ScatterGather, Sequence, Version

# Below this many exploded elements the polars explode+gather (about ten lazy collects, each with
# Python/Rust round-trip overhead) costs more than the per-element Python loop it replaces.
_SCATTER_ELEMENTS = 1024
_PID = "__decider_scatter_pid__"


class _Scope:
    """The rows a node runs on, the frame a frame step sees on them, and the names in sight."""

    __slots__ = ("rows", "base", "names", "arm", "iteration")

    def __init__(self, rows: np.ndarray | None, base: pl.DataFrame, names: dict[str, Version],
                 arm: int | None = None, iteration: int | None = None):
        self.rows = rows
        self.base = base
        self.names = names
        self.arm = arm
        self.iteration = iteration

    def count(self, n: int) -> int:
        return n if self.rows is None else len(self.rows)

    def child(self, keep: np.ndarray | None = None) -> _Scope:
        if keep is None:
            return _Scope(self.rows, self.base, dict(self.names), self.arm, self.iteration)
        rows = np.flatnonzero(keep) if self.rows is None else self.rows[keep]
        base = self.base.filter(pl.Series(keep)) if self.base.width else self.base
        return _Scope(rows, base, dict(self.names), self.arm, self.iteration)

    def checkpoint(self, origin, when: str) -> Checkpoint:
        return Checkpoint(origin, when, self.arm, self.iteration)


class InterpretedRunner:
    """Runs every node in plain Python, row by row; row nodes call their Python `reference`.

    `visit` is called with each locator a row node's `reference` reports (a
    tree's internal nodes); set it to watch them.

    Example::

        for checkpoint in InterpretedRunner().iterate(plan, state, params):
            ...
    """

    # Nodes to pass over as already run, each with the versions it passes on; their values are in the
    # state already. A debug session sets it while it replays a run up to an edit.
    skip: dict[Resolved, list[Version]] = {}

    def __init__(self) -> None:
        self.visit: Callable[[str], None] = _ignore
        self._trace = None
        self._trace_table = None

    def iterate(self, plan: Plan, state: State, params: RunParams, trace=None, table=None) -> Iterator[Checkpoint]:
        self._trace = trace
        self._trace_table = table
        try:
            root = _Scope(None, state.frame, {})
            yield from self._node(plan.root, state, params, root)
            state.frame = root.base
        finally:
            self._trace = None

    def _node(self, r: Resolved, state: State, params: RunParams, scope: _Scope) -> Iterator[Checkpoint]:
        if self.skip and (passed := self.skip.get(r)) is not None:
            scope.names.update((v.name, v) for v in passed)
            return
        origin = r.node.origin
        yield scope.checkpoint(origin, "before")
        if isinstance(r, Call):
            self._call(r, state, params, scope)
        elif isinstance(r, Sequence):
            yield from self._sequence(r, state, params, scope)
        elif isinstance(r, Branch):
            yield from self._branch(r, state, params, scope)
        elif isinstance(r, Loop):
            yield from self._loop(r, state, params, scope)
        else:
            yield from self._scatter_gather(r, state, params, scope)
        yield scope.checkpoint(origin, "after")

    def _sequence(self, seq: Sequence, state: State, params: RunParams, scope: _Scope) -> Iterator[Checkpoint]:
        for child in seq.children:
            yield from self._node(child, state, params, scope)

    def _call(self, call: Call, state: State, params: RunParams, scope: _Scope) -> None:
        node = call.node
        if node.kind == "frame":
            return _frame(call, state, scope, params.bundle(call.id, scope.count(state.n)), self._trace)
        m = scope.count(state.n)
        bundle = params.bundle(call.id, m)
        # Plain Python scalars, not numpy ones: `x / 0.0` must raise here as it does in a kernel.
        cols = [_argument(state, v, i, scope.rows, node.origin.path, node.kind).tolist()
                for i, v in zip(node.inputs, call.reads)]
        rows = zip(*cols) if cols else repeat((), m)
        results: list = []
        append = results.append
        try:
            if node.kind == "scalar":
                args = [i.arg for i in node.inputs]
                fixed = {**dict(node.consts), **{d.arg: b for d, b in zip(node.params, bundle)}}
                for row in rows:
                    append(node.fn(**dict(zip(args, row)), **fixed))
            else:
                consts = tuple(v for _, v in node.consts)
                if node.reference is not None:
                    for row in rows:
                        append(node.reference(row, bundle, consts, self.visit))
                else:
                    for row in rows:
                        append(node.fn(row, bundle, consts))
        except Exception as e:
            k = len(results)
            _note(e, f"in step {node.origin.path}, row {k if scope.rows is None else int(scope.rows[k])}")
            raise
        if node.kind == "scalar" and len(node.outputs) == 1:
            results = [(r,) for r in results]
        columns = list(zip(*results)) if results else [()] * len(node.outputs)
        if len(columns) != len(node.outputs):
            raise ValueError(f"{node.origin.path}: returned {len(columns)} values per row, "
                             f"but declares {len(node.outputs)} outputs")
        for v, out, values in zip(call.writes, node.outputs, columns):
            # A `Struct[Item]` step returning a tuple of fields is stored as the dict a reader expects,
            # so interpreted and compiled modes agree on the value a struct step reads back.
            if (item := struct_item(out.annotation)) is not None:
                fields = [name for name, _ in item_schema(item)]
                values = tuple(None if x is None else (x if isinstance(x, dict) else dict(zip(fields, x)))
                               for x in values)
            # A `Raw[...]` output is stored as a kernel stores it, or the modes disagree on dtype.
            array, valid = _array(values, numpy_dtype(out.annotation) if is_raw(out.annotation)
                                  else dtype_of(base_annotation(out.annotation)))
            state.write(v, array, scope.rows, valid)
            scope.names[v.name] = v
        self._emit_rows(Kind.STEP, node.origin, scope, m, step=call.id + 1)

    def _emit_rows(self, kind: Kind, origin, scope: _Scope, m: int, *, step: int = 0,
                   value: int | float | None = None) -> None:
        trace = self._trace
        if trace is None:
            return
        rows = scope.rows
        for r in range(m):
            trace.emit(kind, origin, step=step, arm=scope.arm, iteration=scope.iteration,
                       record=r if rows is None else int(rows[r]), value=value)

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
            inner.arm = k
            self._emit_rows(Kind.BRANCH_ARM, branch.node.origin, inner, len(inner.rows))
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
        for i in range(1, loop.node.max_iterations + 1):
            active.iteration = i
            self._emit_rows(Kind.LOOP_ITERATION, loop.node.origin, active, active.count(state.n))
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

    def _scatter_gather(self, sg: ScatterGather, state: State, params: RunParams, scope: _Scope) -> Iterator[Checkpoint]:
        node = sg.node
        src = state.source(sg.column) if scope.rows is None else None
        if (src is not None and len(src) > 1 and isinstance(src.dtype, pl.List)
                and isinstance(src.dtype.inner, pl.Struct)
                and (src.list.len().sum() or 0) >= _SCATTER_ELEMENTS):
            yield from self._scatter_gather_vectorized(sg, state, params, scope, src)
            return
        lists, _ = state.read(sg.column, scope.rows)
        names = [name for name, _ in sg.fields]
        # Explode the lists (one per row in scope) into flat per-field element arrays.
        offsets = [0]
        flat = {name: [] for name in names}
        for row_list in lists:
            items = row_list or ()
            for item in items:
                for name in names:
                    flat[name].append(item[name])
            offsets.append(offsets[-1] + len(items))
        elem_state = State(state.plan, pl.DataFrame(), offsets[-1])
        for (name, v) in sg.fields:
            values, valid = _array(tuple(flat[name]), dtype_of(base_annotation(v.annotation)))
            elem_state.write(v, values, valid=valid)
        elem_scope = _Scope(None, elem_state.frame, {name: v for name, v in sg.fields})
        # A child doc forwards the hoisted params from this node's path into the body's nested paths.
        child_params = _forward_params(params, node.origin.path, node.hoist) if node.params else params
        yield from self._node(sg.body, elem_state, child_params, elem_scope)
        # Gather the body's writes back into a list column, one enriched list per row.
        gathered = {v.name: (elem_state.read(v)[0].tolist(), elem_state.read(v)[1]) for v in sg.new_fields}
        out_lists = np.empty(len(lists), object)
        for r, row_list in enumerate(lists):
            items = row_list or ()
            lo = offsets[r]
            out = []
            for j, item in enumerate(items):
                enriched = dict(item)
                for v in sg.new_fields:
                    values, valid = gathered[v.name]
                    k = lo + j
                    enriched[v.name] = None if valid is not None and not valid[k] else record_value(values[k], v.annotation)
                out.append(enriched)
            out_lists[r] = out
        state.write(sg.out, out_lists, scope.rows)
        scope.names[sg.out.name] = sg.out

    def _scatter_gather_vectorized(self, sg: ScatterGather, state: State, params: RunParams,
                                   scope: _Scope, src: pl.Series) -> Iterator[Checkpoint]:
        node = sg.node
        col = src.name
        idx = src.to_frame().with_row_index(_PID)
        exploded = (idx.filter(pl.col(col).list.len().fill_null(0) > 0)
                    .explode(col).unnest(col))
        elem_state = State(state.plan, pl.DataFrame(), exploded.height)
        for (name, v) in sg.fields:
            values, valid = from_series(exploded.get_column(name))
            dtype = dtype_of(base_annotation(v.annotation))
            if values.dtype != dtype:
                values = values.astype(dtype)
            elem_state.write(v, values, valid=valid)
        elem_scope = _Scope(None, elem_state.frame, {name: v for name, v in sg.fields})
        child_params = _forward_params(params, node.origin.path, node.hoist) if node.params else params
        yield from self._node(sg.body, elem_state, child_params, elem_scope)
        # Gather: one enriched struct per element, grouped back into a list per parent row.
        result = exploded.with_columns([elem_state.column(v.name, v) for v in sg.new_fields])
        struct = pl.struct([c for c in result.columns if c != _PID])
        grouped = (result.select(_PID, struct.alias(col)).group_by(_PID, maintain_order=True)
                   .agg(pl.col(col)))
        joined = idx.select(_PID).join(grouped, on=_PID, how="left")
        series = joined.get_column(col).fill_null([])
        lists = series.to_list()
        out_lists = np.empty(len(lists), object)
        for i, row in enumerate(lists):
            out_lists[i] = row
        state.write(sg.out, out_lists, scope.rows)
        # Keep the gathered Series too, so downstream reads (the output frame, a `Columnar[Item]`
        # step) use its Arrow buffers instead of rebuilding them from the Python objects.
        state._sources[sg.out.id] = (out_lists, series)
        scope.names[sg.out.name] = sg.out


def _ignore(locator: str) -> None:
    pass


def _forward_params(params: RunParams, node_path: str, hoist: tuple[tuple[str, str], ...]) -> RunParams:
    # A child body's steps read params at their own nested paths; hoisting surfaces them at the
    # node's path in the document, so forward each hoisted value down into the child path it feeds.
    local: Any = params.doc
    for part in node_path.split("/"):
        local = local.get(part, {}) if isinstance(local, Mapping) else {}
    if not isinstance(local, Mapping) or not local:
        return params
    doc = copy.deepcopy(dict(params.doc))
    for name, child_path in hoist:
        if name not in local:
            continue
        target: Any = doc
        for part in child_path.split("/"):
            target = target.setdefault(part, {})
        target[name] = local[name]
    return RunParams(params.nodes, doc, params.cache, params.lazy)


def _note(e: BaseException, text: str) -> None:
    # The error keeps its type for `except`; the note shows under its traceback (Python 3.11+ only).
    if hasattr(e, "add_note"):
        e.add_note(text)


def _argument(state: State, version: Version, decl: Input, rows: np.ndarray | None, path: str,
              node_kind: str = "scalar") -> np.ndarray:
    values, valid = state.read(version, None)
    span = False
    item = columnar_item(decl.annotation) if values.dtype == object else None
    if values.dtype == object:
        if item is not None:
            schema = item_schema(item)
            values = state.representation(version, ("rows", schema),
                                          lambda v, s, a: build_rows(v, schema, s, a))
        elif base_annotation(decl.annotation) in (str, bytes):
            # Never the row-only span/code shape here: interpreted trees keep real str/bytes values.
            kind = representation_for(decl.annotation, row=False)
            # A `Span` answers what a compiled span answers, so both modes agree.
            span = kind is Representation.RAW_BYTES
            if span:
                values = state.representation(version, (kind, "objects"),
                                              lambda v, s, a: _spans(decl.name, v))
            elif kind is Representation.RAW_STRING:
                values = state.representation(version, kind, lambda v, s, a: codes(v))
            elif kind is Representation.SEMANTIC_BYTES and node_kind == "scalar":
                # `bytes` declares how a step reads a string column: as its UTF-8 bytes. A tree
                # matches on the strings themselves, so only a scalar step is given the bytes.
                values = state.representation(version, (kind, "encoded"),
                                              lambda v, s, a: _encoded(decl.name, v))
    if rows is not None:
        values = values[rows]
        valid = None if valid is None else valid[rows]
    if valid is None or valid.all():
        return values
    missing = ~valid
    if decl.null_policy is NullPolicy.REQUIRED:
        raise MissingInputError(decl.name, path, int(missing.sum()), len(valid), absent=_absent(state, version))
    if span:
        if decl.null_policy is NullPolicy.MISSING_AS:
            return span_objects([s.text for s in values], decl.fill, missing)
        # A null span carries its own -1 length, which None would lose.
        return values
    if decl.null_policy is NullPolicy.MISSING_AS:
        # A null row of a `Columnar[...]` input already reads as a row with no items.
        return values if rows_needs_no_fill(decl, item is not None) else fill_missing(values, valid, decl.fill)
    values = values.astype(object)
    values[missing] = None
    return values


def _encoded(name: str, values: np.ndarray) -> np.ndarray:
    try:
        return np.array([v if v is None or isinstance(v, bytes) else str.encode(v) for v in values], object)
    except TypeError as e:
        raise TypeError(f"'{name}' is a string input: {e}") from None


def _spans(name: str, values: np.ndarray) -> np.ndarray:
    try:
        return span_objects(values)
    except TypeError as e:
        raise TypeError(f"'{name}' is a string input: {e}") from None


def _absent(state: State, version: Version) -> bool:
    # A state stores every input column it was given, so one never written was absent.
    return version.producer is None and version.id not in state.values


def _array(values: tuple, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray | None]:
    # A step returning None writes a null, which later readers see through their null policy.
    valid = np.array([x is not None for x in values], bool)
    if valid.all():
        valid = None
    elif dtype != object:
        values = tuple(0 if x is None else x for x in values)
    if dtype != object:
        return np.array(values, dtype), valid
    # Filled one by one so a tuple or list value stays one element.
    out = np.empty(len(values), object)
    for i, v in enumerate(values):
        out[i] = v
    return out, valid


def _frame(call: Call, state: State, scope: _Scope, bundle: tuple, trace=None) -> None:
    node = call.node
    path = node.origin.path
    df = state.frame_of(scope.base, scope.names, scope.rows)
    try:
        out = node.fn(df, **{d.arg: b for d, b in zip(node.params, bundle)})
    except Exception as e:
        _note(e, f"in frame step {path}")
        raise
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
        # A nested column a frame step writes keeps its Series, so a later `Columnar[Item]` read can
        # use its Arrow buffers instead of rebuilding the flat arrays from Python objects.
        state._sources[v.id] = (values, out[v.name])
    if node.outputs is None:
        # Unknown lineage: its frame is all that later nodes see.
        scope.base, scope.names = out, {}
    else:
        scope.names.update((v.name, v) for v in call.writes)
    if trace is not None:
        rows = scope.rows
        m = len(rows) if rows is not None else state.n
        for r in range(m):
            trace.emit(Kind.FRAME, node.origin, step=call.id + 1,
                       record=r if rows is None else int(rows[r]))
