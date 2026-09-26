from __future__ import annotations

import threading
from typing import Iterator

import numpy as np
import polars as pl

from decider.engine.boundary.nulls import MissingInputError
from decider.engine.compile import Fallback, Unit, compile_plan, numpy_dtype
from decider.engine.ir.decls import Input, NullPolicy, base_annotation
from decider.engine.run.params import RunParams
from decider.engine.run.runners.base import Checkpoint
from decider.engine.run.runners.interpreted import InterpretedRunner, _absent, _note, _Scope
from decider.engine.run.state import State
from decider.engine.wiring.plan import Call, Plan, Version


class SteppedRunner(InterpretedRunner):
    """Runs every scalar and row node as its own numba kernel, with a Python driver in between.

    Pauses at every node, like the interpreted runner. Frame steps, branches
    and loops run in Python. A `str` value enters a kernel as an int32 code
    and a `str` param as the code of its literal, so a step compares a `str`
    input against a `str` param exactly as it does in plain Python. A `bytes`
    input enters as an `(address, byte length)` span of its UTF-8 bytes, and
    then so does each `str` param of that node. A Python fallback gets plain
    Python values.

    Example::

        exe = Engine().bind(pipeline, mode="stepped")
        for checkpoint in exe.runner.iterate(exe.plan, *exe.prepare(df)):
            ...
    """

    fuse = False

    def __init__(self) -> None:
        super().__init__()
        self._plan: Plan | None = None
        self.units: dict[int, Unit] = {}
        self._reads: dict[int, tuple[tuple[Input, Version, str], ...]] = {}
        self._strs: dict[int, tuple[str, ...]] = {}
        self._converted: dict[tuple[str, int], tuple] = {}
        self._codes: dict[str, int] = {}
        self._spans: set[int] = set()
        self._alive: dict[tuple[str, int], dict] = {}
        self._lock = threading.Lock()

    def iterate(self, plan: Plan, state: State, params: RunParams) -> Iterator[Checkpoint]:
        # Not a generator itself: one generator frame less per checkpoint on the single-record path.
        if plan is not self._plan:
            self._compile(plan, params.lazy)
        return super().iterate(plan, state, params)

    def _compile(self, plan: Plan, lazy: bool) -> None:
        strs = {c.id: _str_params(c) for c in plan.calls if c.node.kind == "scalar"}
        # A row node reading `bytes` compares bytes, so its `str` params go in as UTF-8 spans.
        self._spans = {c.id for c in plan.calls if c.node.kind == "row" and _reads_bytes(c)}
        strs |= {k: tuple(d.name for d in c.node.params if d.annotation is str)
                 for c in plan.calls if (k := c.id) in self._spans}
        self.units = compile_plan(plan, fuse=self.fuse)
        self._reads = {id(unit): _external(unit) for unit in self.units.values()}
        self._strs = {k: v for k, v in strs.items() if v}
        # Converted bundles are keyed by call id, which means another call in another plan.
        self._converted, self._alive = {}, {}
        self._plan = plan

    def _call(self, call: Call, state: State, params: RunParams, scope: _Scope) -> None:
        unit = self.units.get(call.id)
        if unit is None:
            return super()._call(call, state, params, scope)
        self._run(unit, state, params, scope)

    def _run(self, unit: Unit, state: State, params: RunParams, scope: _Scope) -> None:
        rows = scope.rows
        n = scope.count(state.n)
        # A fallback runs Python code, which takes Python values: strings, not codes.
        python = isinstance(unit, Fallback)
        # Every call of a unit runs on every row the unit runs on, so validating
        # them all here is validating exactly the nodes that run.
        bundles = {c.id: params.bundle(c.id, n) if python else self._bundle(c.id, params, n)
                   for c in unit.calls if c.node.params}
        values: dict[int, np.ndarray] = {}
        valid: dict[int, np.ndarray] = {}
        alive: list = []
        for decl, v, path in self._reads[id(unit)]:
            x, mask = state.read(v, rows)
            if x.dtype == object and not python:
                x = self._typed(x, mask, decl, alive, None if rows is not None else state.source(v))
            if mask is not None and not mask.all():
                if decl.null_policy is NullPolicy.REQUIRED:
                    raise MissingInputError(decl.name, path, int((~mask).sum()), len(mask), absent=_absent(state, v))
                if decl.null_policy is NullPolicy.MISSING_AS:
                    # ponytail: one fill per version per kernel; two readers with different fills share the first.
                    x = np.where(mask, x, decl.fill).astype(x.dtype)
                valid[v.id] = mask
            values.setdefault(v.id, x)
        try:
            unit.run(values, valid, bundles, n)
        except Exception as e:
            paths = [c.node.origin.path for c in unit.calls]
            _note(e, f"in step {paths[0]}" if len(paths) == 1 else
                  f"in one of the steps {paths}, fused into one kernel; run in mode='stepped' to see which")
            raise
        for v, _ in unit.writes:
            state.write(v, values[v.id], rows, valid.get(v.id))
            scope.names[v.name] = v

    def _typed(self, x: np.ndarray, mask: np.ndarray | None, decl: Input, alive: list,
               source: pl.Series | None) -> np.ndarray:
        if base_annotation(decl.annotation) is bytes:
            try:
                return _spans(x, mask, alive, source)
            except TypeError as e:
                raise TypeError(f"'{decl.name}' is a string input: {e}") from None
        if base_annotation(decl.annotation) is str:
            get = self._codes.get
            return np.fromiter((get(s, -1) for s in x), np.int32, len(x))
        if mask is not None:
            x = np.where(mask, x, 0)
        return x.astype(numpy_dtype(base_annotation(decl.annotation)))

    def _bundle(self, call_id: int, params: RunParams, n: int) -> tuple:
        bundle = params.bundle(call_id, n)
        names = self._strs.get(call_id)
        if names is None:
            return bundle
        key = (params.key, call_id)
        converted = self._converted.get(key)
        if converted is None:
            if call_id in self._spans:
                utf8 = {k: np.frombuffer(getattr(bundle, k).encode(), np.uint8) for k in names}
                # Kept with the cached bundle, whose spans point into them.
                self._alive[key] = utf8
                codes = {k: (b.ctypes.data, len(b)) for k, b in utf8.items()}
            else:
                with self._lock:
                    codes = {k: np.int32(self._codes.setdefault(getattr(bundle, k), len(self._codes))) for k in names}
            converted = self._converted[key] = bundle._replace(**codes)
        return converted


def _external(unit: Unit) -> tuple[tuple[Input, Version, str], ...]:
    # What a unit reads from outside itself, with the null policy that applies.
    inside = {v.id for c in unit.calls for v in c.writes} | getattr(unit, "inner", set())
    reads = [(i, v, c.node.origin.path) for c in unit.calls for i, v in zip(c.node.inputs, c.reads) if v.id not in inside]
    # A value a packed branch or loop only copies needs no policy: a null in it sends the run down the unpacked path.
    reads += [(Input(v.name, base_annotation(v.annotation)), v, "") for v in getattr(unit, "passthrough", ())]
    return tuple(reads)


def _reads_bytes(call: Call) -> bool:
    return any(base_annotation(i.annotation) is bytes for i in call.node.inputs)


def _spans(x: np.ndarray, mask: np.ndarray | None, alive: list, source: pl.Series | None) -> np.ndarray:
    # Strings reach a kernel as `(address, byte length)` spans into Arrow memory, -1 for a null.
    from decider.engine.boundary.extract import extract_frame

    if mask is not None:
        x = np.where(mask, x, None)
    if len(x) <= 32:
        # A few records (score): encoding them beats building a frame to export.
        raw = [None if s is None else str.encode(s) for s in x]
        buffer = np.frombuffer(b"".join(b for b in raw if b) or bytes(1), np.uint8)
        alive.append(buffer)
        lengths = np.array([-1 if b is None else len(b) for b in raw], np.int64)
        starts = np.cumsum(np.maximum(lengths, 0)) - np.maximum(lengths, 0)
        return np.stack([buffer.ctypes.data + starts, lengths], axis=1)
    # The input frame's own column when it holds these values, else a copy (an override, a row subset).
    if source is None or source.dtype != pl.String:
        source = pl.Series(x.tolist(), dtype=pl.String)
    extracted = extract_frame(source.to_frame("s"), [Input("s", bytes, NullPolicy.OPTIONAL)])
    alive.append(extracted.kernel_frame)
    return extracted.columns["s"].values


def _str_params(call: Call) -> tuple[str, ...]:
    node = call.node
    strs = [i.name for i in node.inputs if base_annotation(i.annotation) is str]
    params = tuple(d.name for d in node.params if d.annotation is str)
    path = node.origin.path
    if not strs:
        if params:
            raise ValueError(
                f"{path}: `str` param '{params[0]}' reaches a compiled kernel as the code of its literal, "
                "which only means something compared with a `str` input, and this step reads none"
            )
        return params
    if len(strs) > 1:
        raise ValueError(
            f"{path}: reads several `str` inputs {strs}; compiled modes compare a `str` input only "
            "with a `str` param, so split the step or run it in interpreted mode"
        )
    if not params or any(isinstance(v, str) for _, v in node.consts):
        raise ValueError(
            f"{path}: `str` input '{strs[0]}' enters a compiled kernel as a code, so a literal in the "
            "function body would never match it; declare the literal as a `str` param, "
            "e.g. `private: str = param(\"private\")`"
        )
    return params
