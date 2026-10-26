from __future__ import annotations

import threading
import warnings
from functools import partial
from typing import Any, Iterable, Iterator, NamedTuple

import numpy as np
import polars as pl
from numba.core.dispatcher import Dispatcher

from decikit._engine.boundary.nulls import MissingInputError
from decikit._engine.compile import Fallback, Kernel, Packed, Unit, compile_plan, numpy_dtype
from decikit._engine.compile.kernel import NOT_WHOLE
from decikit._engine.compile.rows import build_ragged, build_rows, rows_needs_no_fill
from decikit._engine.compile.structs import build_struct, struct_schema
from decikit._engine.ir.decls import Input, NullPolicy, base_annotation, is_typed, stored_fill
from decikit._engine.run.params import RunParams
from decikit._engine.run.representations import span_objects, spans
from decikit._engine.run.runners.base import Checkpoint
from decikit._engine.run.runners.interpreted import InterpretedRunner, _absent, _Context, _encoded, _note, _Scope
from decikit._engine.run.state import State, all_set, fill_missing
from decikit._engine.trace.envelope import Kind
from decikit.exceptions import FallbackWarning, NotIntegralError, StrictCompileError
from decikit.ops.nested import struct_item
from decikit._engine.types import Representation, is_raw, representation_for, columnar_item, item_schema
from decikit._engine.wiring.plan import Call, Plan, NodeVersion


# Everything a runner compiles for one plan, published whole so a call never sees half of it.
class Compiled(NamedTuple):
    plan: Plan | None
    units: dict[int, Unit]
    reads: dict[int, tuple[tuple[Input, NodeVersion, str, np.dtype | None, Any, Any, bool], ...]]
    strs: dict[int, tuple[str, ...]]
    fallbacks: dict[str, str]
    python: dict[int, str]
    # Keyed by call id, which means another call in another plan. Entries are written once and never
    # replaced: `(bundle, utf8)`, the bundle's spans pointing into the buffers beside it.
    converted: dict[tuple[str, int], tuple]
    packed: dict[str, Packed]


class SteppedRunner(InterpretedRunner):
    """Runs every scalar and row node as its own numba kernel, with a Python driver in between.

    Pauses at every node, like the interpreted runner. Frame steps, branches
    and loops run in Python. A `Text` or `bytes` input enters as
    an `(address, byte length)` span of its UTF-8 bytes, and then so does each
    `str` param of that node. A step running in Python gets plain Python
    values, except a `Text` or `Columnar[...]` input, whose representation is
    part of what it declares.

    A step no kernel holds -- one reading or writing a semantic `str`, a `date`,
    a `list` or a `dict`, or with a body numba can't compile -- runs in Python,
    row by row, and so does every step of a kernel numba refuses only when it
    is launched. `fallbacks()` lists each, without a warning; with
    `strict=True` it raises instead, unless the step is `@allow_fallback`. A
    `Columnar[Item]` input still runs compiled, one call per row.

    Example::

        exe = SteppedEngine().bind(pipeline)
        for checkpoint in exe.runner.iterate(exe.plan, *exe.prepare(df)):
            ...
    """

    fuse = False
    # Whether a step outside a shared kernel warns with a `FallbackWarning`; `strict` refuses one either way.
    loud = False

    def __init__(self, strict: bool = False) -> None:
        super().__init__()
        self.strict = strict
        self._lock = threading.Lock()
        self._compiled = Compiled(None, {}, {}, {}, {}, {}, {}, {})

    @property
    def _plan(self) -> Plan | None:
        return self._compiled.plan

    @property
    def units(self) -> dict[int, Unit]:
        return self._compiled.units

    @property
    def packed(self) -> dict[str, Packed]:
        return self._compiled.packed

    def iterate(self, plan: Plan, state: State, params: RunParams, trace=None, table=None,
                quiet: bool = False) -> Iterator[Checkpoint]:
        compiled = self._compiled
        if compiled.plan is not plan:
            compiled = self._compile(plan, params.lazy)
        return self._drive(state, params, _Context(trace, table, compiled, quiet), plan)

    def _compile(self, plan: Plan, lazy: bool) -> Compiled:
        with self._lock:
            compiled = self._compiled
            if compiled.plan is not plan:
                compiled = self._compiled = self._build(plan, lazy)
            return compiled

    def _build(self, plan: Plan, lazy: bool) -> Compiled:
        python: dict[int, str] = {}
        # A call reading `bytes`, or a `Columnar[...]` item's `str` field, compares spans, so its
        # `str` params go in as UTF-8 spans too.
        strs = {c.id: names for c in plan.calls if _reads_span(c)
                and (names := tuple(d.name for d in c.node.params if d.annotation is str))}
        units = compile_plan(plan, fuse=self.fuse, python=python)
        fallbacks = self._report(u for u in units.values() if isinstance(u, Fallback))
        for unit in units.values():
            if isinstance(unit, Kernel):
                unit.on_split = partial(self._split, plan)
        reads = {id(unit): _external(unit) for unit in units.values()}
        return Compiled(plan, units, reads, strs, fallbacks, python, {}, {})

    def _report(self, units: Iterable[Fallback]) -> dict[str, str]:
        # Every step outside a shared kernel, by path; strict refuses one that runs Python undeclared.
        fallbacks: dict[str, str] = {}
        for unit in units:
            # A fallback backed by a dispatcher still runs compiled, just one call per row.
            compiled = isinstance(unit.fn, Dispatcher)
            for c in unit.calls:
                path = c.node.origin.path
                fallbacks[path] = f"@allow_fallback: {unit.reason}" if unit.declared else unit.reason
                if unit.declared:
                    continue
                if compiled:
                    if self.loud:
                        warnings.warn(f"{path} runs compiled, one call per row outside the shared "
                                      f"kernel: {unit.reason}", FallbackWarning, stacklevel=6)
                    continue
                message = f"{path} runs in Python, row by row: {unit.reason}"
                if self.strict:
                    raise StrictCompileError(f"{message}; accept it with @allow_fallback, or run engine='interpreted'")
                if self.loud:
                    warnings.warn(message, FallbackWarning, stacklevel=6)
        return fallbacks

    def _split(self, plan: Plan, units: tuple[Fallback, ...]) -> None:
        # A kernel splitting at a launch after the build: reported like a build-time fallback, once.
        with self._lock:
            now = self._compiled
            if now.plan is plan and units[0].calls[0].node.origin.path in now.fallbacks:
                return   # another thread's launch split it first
            found = self._report(units)
            if now.plan is plan:
                self._compiled = now._replace(fallbacks=now.fallbacks | found)

    def fallbacks(self) -> dict[str, str]:
        return dict(self._compiled.fallbacks)

    def _call(self, call: Call, state: State, params: RunParams, scope: _Scope) -> None:
        unit = scope.ctx.compiled.units.get(call.id)
        if unit is None:
            return super()._call(call, state, params, scope)
        self._run(unit, state, params, scope)

    def _run(self, unit: Unit, state: State, params: RunParams, scope: _Scope) -> None:
        rows = scope.rows
        n = scope.count(state.n)
        # A genuinely interpreted fallback runs Python code, which takes Python values: real strings,
        # not codes. A fallback backed by a compiled dispatcher (`Columnar[Item]`) still wants
        # typed/representation values, same as a `Kernel`.
        python = isinstance(unit, Fallback) and not isinstance(unit.fn, Dispatcher)
        # Every call of a unit runs on every row the unit runs on, so validating
        # them all here is validating exactly the nodes that run.
        ctx = scope.ctx
        compiled = ctx.compiled
        bundles = {c.id: params.bundle(c.id, n) if python else self._bundle(compiled, c.id, params, n)
                   for c in unit.calls if c.node.params}
        values: dict[int, np.ndarray] = {}
        valid: dict[int, np.ndarray] = {}
        alive: list = []
        for decl, v, path, want, kind, build, raw in compiled.reads[id(unit)]:
            x, mask = state.read(v, rows)
            filled = False
            # `Text` and `Columnar[...]` are a contract about the value, not an optimisation, so
            # their representation is built for a step running in Python too, where every other
            # annotation wants the Python value instead.
            if _records(kind, python) or (x.dtype == object and (raw or not python)):
                if kind is None:
                    # A boxed number (a `missing_as` fill, an input absent from the frame): just cast it.
                    if mask is not None:
                        x = np.where(mask, x, 0)
                    x = x.astype(build)
                elif _fills(decl, mask, kind):
                    # The fill belongs in the strings: a span or a code cannot be filled afterwards,
                    # and the fill is this reader's, not the shared representation's.
                    filled = True
                    x = build(fill_missing(x, mask, stored_fill(decl)), None, alive)
                elif not _null_struct(kind, mask):
                    full = state.representation(v, kind, build)
                    x = full if rows is None else full[rows]
            elif not python and want is not None and x.dtype != want and np.can_cast(x.dtype, want):
                # An int column read as `float` (another step reads it as `int`): a kernel types what it gets.
                x = x.astype(want)
            if mask is not None and not all_set(mask):
                if decl.null_policy is NullPolicy.REQUIRED:
                    raise MissingInputError(decl.name, path, int((~mask).sum()), len(mask), absent=_absent(state, v))
                if not filled and _fills(decl, mask, kind):
                    # ponytail: one fill per version per kernel; two readers with different fills share the first.
                    # A struct's records come back as dicts beside the fill, as a Python body takes them.
                    x = fill_missing(x, mask, stored_fill(decl)).astype(object if x.dtype.names else x.dtype, copy=False)
                valid[v.id] = mask
            values.setdefault(v.id, x)
        try:
            trace = ctx.trace
            # In-kernel capture only at a plain top-level scope: a packed
            # branch or loop carries its own arm/iteration inside the kernel,
            # but a single-call kernel inside an arm or iteration has neither,
            # and a subset of rows would mislabel records from the buffer's
            # row-major offsets.
            top = scope.rows is None and scope.arm is None and scope.iteration is None
            captured = (unit.run(values, valid, bundles, n, trace, ctx.table)
                        if trace is not None and isinstance(unit, Kernel) and top else unit.run(values, valid, bundles, n))
            if trace is not None and not captured:
                for c in unit.calls:
                    self._emit_rows(Kind.STEP, c.node.origin, scope, n, step=c.id + 1)
        except Exception as e:
            paths = [c.node.origin.path for c in unit.calls]
            where = (f"step {paths[0]}" if len(paths) == 1 else
                     f"one of the steps {paths}, fused into one kernel; run in engine='stepped' to see which")
            if isinstance(e, NotIntegralError) and e.message == NOT_WHOLE:
                # The kernel is shared by content, so its message can't name the step.
                raise NotIntegralError(f"in {where}: {e.message}") from None
            _note(e, f"in {where}")
            raise
        for v, _ in unit.writes:
            state.write(v, values[v.id], rows, valid.get(v.id))
            scope.names[v.name] = v

    def _bundle(self, compiled: Compiled, call_id: int, params: RunParams, n: int) -> tuple:
        bundle = params.bundle(call_id, n)
        names = compiled.strs.get(call_id)
        if names is None:
            return bundle
        key = (params.key, call_id)
        entry = compiled.converted.get(key)
        if entry is None:
            utf8 = {k: np.frombuffer(getattr(bundle, k).encode(), np.uint8) for k in names}
            pointers = {k: (b.ctypes.data, len(b)) for k, b in utf8.items()}
            # A thread that lost the race uses the winner's entry, so the buffers its spans point
            # into are never freed under a running kernel.
            entry = compiled.converted.setdefault(key, (bundle._replace(**pointers), utf8))
        return entry[0]


def _external(unit: Unit) -> tuple[tuple[Input, NodeVersion, str, np.dtype | None, Any, Any, bool], ...]:
    # What a unit reads from outside itself, with the null policy that applies, the dtype a number is
    # read as, and how an object array of it becomes what the kernel takes. Everything here is decided
    # once per plan: a single record has no time for annotation introspection per call.
    inside = {v.id for c in unit.calls for v in c.writes} | getattr(unit, "inner", set())
    reads = [(i, v, c.node.origin.path) for c in unit.calls for i, v in zip(c.node.inputs, c.reads) if v.id not in inside]
    # A value a packed branch or loop only copies needs no policy: a null in it sends the run down the unpacked path.
    reads += [(Input(v.name, base_annotation(v.annotation)), v, "") for v in getattr(unit, "passthrough", ())]
    row = any(c.node.kind == "row" for c in unit.calls)
    # Only a shared array kernel takes the flat arrays; a fallback is called once per row and
    # wants that row's items already sliced.
    kernel = not isinstance(unit, Fallback)
    # A genuinely interpreted fallback runs a Python body, which compares whole values, so it needs
    # `Text` objects; a compiled body reads the `(address, length)` pairs through span overloads.
    python = isinstance(unit, Fallback) and not isinstance(unit.fn, Dispatcher)
    return tuple((i, v, path, numpy_dtype(b) if is_typed(b := base_annotation(i.annotation)) else None,
                  *_boxed(i, row, path, kernel, python), _declared(i.annotation, row)) for i, v, path in reads)


def _boxed(decl: Input, row: bool, path: str, kernel: bool = False,
           python: bool = False) -> tuple[Any, Any]:
    # The `(kind, build)` pair `State.representation` needs for an object array of `decl`, or
    # `(None, dtype)` for a boxed number, which is only ever cast.
    item = struct_item(decl.annotation)
    if item is not None:
        fields = struct_schema(item)
        return ("struct", fields), lambda values, source, alive: build_struct(fields, values, source, decl.name, path)
    item = columnar_item(decl.annotation)
    if item is not None:
        schema = item_schema(item)
        # A shared array kernel takes the flat arrays and slices them per row itself; a row node's
        # per-row dispatcher and a per-row `Fallback` still want that row's items already sliced.
        if kernel and not row:
            return ("ragged", schema), lambda values, source, alive: build_ragged(values, schema, source, alive)
        return ("rows", schema), lambda values, source, alive: build_rows(values, schema, source, alive)
    annotation = base_annotation(decl.annotation)
    if annotation not in (str, bytes):
        return None, numpy_dtype(annotation)
    kind = representation_for(decl.annotation, row=row)
    if kind is Representation.RAW_BYTES:
        # The same key interpreted uses, so both share one built representation.
        if python:
            return (kind, "objects"), lambda values, source, alive: span_objects(values)
        return kind, partial(_spans, decl.name)
    if annotation is bytes:
        # A semantic `bytes` step runs in Python and compares whole values, so give it real bytes:
        # the column is strings, and `bytes` declares how this step reads them.
        return kind, lambda values, source, alive: _encoded(decl.name, values)
    # A step reading semantic strings runs one compiled call per row, which takes the Python
    # values as they are: numba types them as its own unicode.
    return kind, lambda values, source, alive: values


def _spans(name: str, values: np.ndarray, source: pl.Series | None, alive: list) -> np.ndarray:
    try:
        return spans(values, None, alive, source)
    except TypeError as e:
        raise TypeError(f"'{name}' is a string input: {e}") from None


def _declared(annotation: Any, row: bool) -> bool:
    # `Text` and `Columnar[...]` say how the value is read, so they hold when the step runs in Python
    # too. A scalar step's `bytes` says the same; a row node falling back matches on the strings.
    return is_raw(annotation) or (not row and base_annotation(annotation) is bytes)


def _records(kind: Any, python: bool) -> bool:
    # A struct column reaches a kernel as a record per row whatever dtype its values arrived in.
    return not python and type(kind) is tuple and kind[0] == "struct"


def _null_struct(kind: Any, mask: np.ndarray | None) -> bool:
    # A null struct belongs to the null policy, which knows whether the column is absent entirely;
    # the record builder would only report it as a null field.
    return type(kind) is tuple and kind[0] == "struct" and mask is not None and not all_set(mask)


def _fills(decl: Input, mask: np.ndarray | None, kind: Any) -> bool:
    if decl.null_policy is not NullPolicy.MISSING_AS or mask is None or all_set(mask):
        return False
    # A null `Columnar[...]` row already reads as a row with no items, which is what an empty fill asks for.
    return not rows_needs_no_fill(decl, type(kind) is tuple and kind[0] in ("rows", "ragged"))


def _reads_span(call: Call) -> bool:
    # A frame node's lineage is unknown, so it has no declared inputs.
    for i in call.node.inputs or ():
        if base_annotation(i.annotation) is bytes:
            return True
        if (item := columnar_item(i.annotation)) is not None and any(
                base_annotation(t) is str for _, t in item_schema(item)):
            return True
    return False
