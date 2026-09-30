from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Iterator

import numpy as np
import polars as pl
from numba.core.dispatcher import Dispatcher

from decider.engine.boundary.nulls import MissingInputError
from decider.engine.compile import Fallback, Unit, compile_plan, numpy_dtype
from decider.engine.compile.rows import build_ragged, build_rows, rows_needs_no_fill
from decider.engine.compile.structs import build_struct, struct_schema
from decider.engine.ir.decls import Input, NullPolicy, base_annotation
from decider.engine.run.params import RunParams
from decider.engine.run.representations import codes, span_objects, spans
from decider.engine.run.runners.base import Checkpoint
from decider.engine.run.runners.interpreted import InterpretedRunner, _absent, _encoded, _note, _Scope
from decider.engine.run.state import State, fill_missing
from decider.exceptions import FallbackWarning
from decider.types import Representation, is_raw, representation_for, columnar_item, item_schema, struct_item
from decider.engine.wiring.plan import Call, Plan, Version


class SteppedRunner(InterpretedRunner):
    """Runs every scalar and row node as its own numba kernel, with a Python driver in between.

    Pauses at every node, like the interpreted runner. Frame steps, branches
    and loops run in Python. A `Raw[str]` value enters a kernel as the int32
    dictionary code `raw_str()` gives that string. A `bytes` input enters as
    an `(address, byte length)` span of its UTF-8 bytes, and then so does each
    `str` param of that node. A step running in Python gets plain Python
    values, except a `Raw[...]` or `Columnar[...]` input, whose representation is
    part of what it declares.

    A step no kernel holds -- one reading or writing a semantic `str`, a `date`
    or a `list`, or with a body numba can't compile -- runs in Python, row by
    row, with one `FallbackWarning` naming it. With `strict=True` it raises
    instead, unless the step is `@allow_fallback`. A `Columnar[Item]` input still
    runs compiled, one call per row.

    Example::

        exe = Engine().bind(pipeline, mode="stepped")
        for checkpoint in exe.runner.iterate(exe.plan, *exe.prepare(df)):
            ...
    """

    fuse = False

    def __init__(self, strict: bool = False) -> None:
        super().__init__()
        self.strict = strict
        self._plan: Plan | None = None
        self.units: dict[int, Unit] = {}
        self._reads: dict[int, tuple[tuple[Input, Version, str, np.dtype | None, Any, Any, bool], ...]] = {}
        self._strs: dict[int, tuple[str, ...]] = {}
        self._converted: dict[tuple[str, int], tuple] = {}
        self._alive: dict[tuple[str, int], dict] = {}
        self._fallbacks: dict[str, str] = {}

    def iterate(self, plan: Plan, state: State, params: RunParams) -> Iterator[Checkpoint]:
        # Not a generator itself: one generator frame less per checkpoint on the single-record path.
        if plan is not self._plan:
            self._compile(plan, params.lazy)
        return super().iterate(plan, state, params)

    def _compile(self, plan: Plan, lazy: bool) -> None:
        python: dict[int, str] = {}
        # A call reading `bytes`, or a `Columnar[...]` item's `str` field, compares spans, so its
        # `str` params go in as UTF-8 spans too.
        strs = {c.id: names for c in plan.calls if _reads_span(c)
                and (names := tuple(d.name for d in c.node.params if d.annotation is str))}
        self.units = compile_plan(plan, fuse=self.fuse, python=python)
        self._fallbacks = {}
        for unit in self.units.values():
            if not isinstance(unit, Fallback):
                continue
            # A fallback backed by a dispatcher still runs compiled, just one call per row.
            compiled = isinstance(unit.fn, Dispatcher)
            for c in unit.calls:
                path = c.node.origin.path
                self._fallbacks[path] = f"@allow_fallback: {unit.reason}" if unit.declared else unit.reason
                if unit.declared:
                    continue
                if compiled:
                    warnings.warn(f"{path} runs compiled, one call per row outside the shared "
                                  f"kernel: {unit.reason}", FallbackWarning, stacklevel=4)
                    continue
                message = f"{path} runs in Python, row by row: {unit.reason}"
                if self.strict:
                    raise ValueError(f"{message}; accept it with @allow_fallback, or run mode='interpreted'")
                warnings.warn(message, FallbackWarning, stacklevel=4)
        self._python = python
        self._reads = {id(unit): _external(unit) for unit in self.units.values()}
        self._strs = strs
        # Converted bundles are keyed by call id, which means another call in another plan.
        self._converted, self._alive = {}, {}
        self._plan = plan

    def fallbacks(self) -> dict[str, str]:
        return dict(self._fallbacks)

    def _call(self, call: Call, state: State, params: RunParams, scope: _Scope) -> None:
        unit = self.units.get(call.id)
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
        bundles = {c.id: params.bundle(c.id, n) if python else self._bundle(c.id, params, n)
                   for c in unit.calls if c.node.params}
        values: dict[int, np.ndarray] = {}
        valid: dict[int, np.ndarray] = {}
        alive: list = []
        for decl, v, path, want, kind, build, raw in self._reads[id(unit)]:
            x, mask = state.read(v, rows)
            filled = False
            # `Raw[...]` and `Columnar[...]` are a contract about the value, not an optimisation, so
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
                    x = build(fill_missing(x, mask, decl.fill), None, alive)
                elif not _null_struct(kind, mask):
                    full = state.representation(v, kind, build)
                    x = full if rows is None else full[rows]
            elif not python and want is not None and x.dtype != want and np.can_cast(x.dtype, want):
                # An int column read as `float` (another step reads it as `int`): a kernel types what it gets.
                x = x.astype(want)
            if mask is not None and not mask.all():
                if decl.null_policy is NullPolicy.REQUIRED:
                    raise MissingInputError(decl.name, path, int((~mask).sum()), len(mask), absent=_absent(state, v))
                if not filled and _fills(decl, mask, kind):
                    # ponytail: one fill per version per kernel; two readers with different fills share the first.
                    x = fill_missing(x, mask, decl.fill).astype(x.dtype, copy=False)
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

    def _bundle(self, call_id: int, params: RunParams, n: int) -> tuple:
        bundle = params.bundle(call_id, n)
        names = self._strs.get(call_id)
        if names is None:
            return bundle
        key = (params.key, call_id)
        converted = self._converted.get(key)
        if converted is None:
            utf8 = {k: np.frombuffer(getattr(bundle, k).encode(), np.uint8) for k in names}
            # Kept with the cached bundle, whose spans point into them.
            self._alive[key] = utf8
            pointers = {k: (b.ctypes.data, len(b)) for k, b in utf8.items()}
            converted = self._converted[key] = bundle._replace(**pointers)
        return converted


def _external(unit: Unit) -> tuple[tuple[Input, Version, str, np.dtype | None, Any, Any, bool], ...]:
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
    # `Span` objects; a compiled body reads the `(address, length)` pairs through span overloads.
    python = isinstance(unit, Fallback) and not isinstance(unit.fn, Dispatcher)
    return tuple((i, v, path, numpy_dtype(b) if (b := base_annotation(i.annotation)) in (float, int, bool) else None,
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
    if kind is Representation.RAW_STRING:
        return kind, lambda values, source, alive: codes(values)
    # A step reading semantic strings runs one compiled call per row, which takes the Python
    # values as they are: numba types them as its own unicode.
    return kind, lambda values, source, alive: values


def _spans(name: str, values: np.ndarray, source: pl.Series | None, alive: list) -> np.ndarray:
    try:
        return spans(values, None, alive, source)
    except TypeError as e:
        raise TypeError(f"'{name}' is a string input: {e}") from None


def _declared(annotation: Any, row: bool) -> bool:
    # `Raw[...]` and `Columnar[...]` say how the value is read, so they hold when the step runs in Python
    # too. A scalar step's `bytes` says the same; a row node falling back matches on the strings.
    return is_raw(annotation) or (not row and base_annotation(annotation) is bytes)


def _records(kind: Any, python: bool) -> bool:
    # A struct column reaches a kernel as a record per row whatever dtype its values arrived in.
    return not python and type(kind) is tuple and kind[0] == "struct"


def _null_struct(kind: Any, mask: np.ndarray | None) -> bool:
    # A null struct belongs to the null policy, which knows whether the column is absent entirely;
    # the record builder would only report it as a null field.
    return type(kind) is tuple and kind[0] == "struct" and mask is not None and not mask.all()


def _fills(decl: Input, mask: np.ndarray | None, kind: Any) -> bool:
    if decl.null_policy is not NullPolicy.MISSING_AS or mask is None or mask.all():
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
