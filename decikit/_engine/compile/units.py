from __future__ import annotations

from typing import Any, Iterator, Mapping, Union

import numpy as np
from numba.core.dispatcher import Dispatcher

from decikit._engine.boundary.whole import refuse
from decikit._engine.compile.declared import check_output
from decikit._engine.compile.kernel import Spec, SrcKind, fused_kernel
from decikit._engine.compile.njit import FALLBACK_ERRORS, compile_call, parameters
from decikit._engine.compile.outputs import CodeTable, coded, coded_values, output_dtype
from decikit._engine.compile.rows import Ragged
from decikit._engine.compile.sink import Sink
from decikit._engine.compile.structs import Schema, to_dicts
from decikit._engine.ir.decls import Input, NullPolicy, base_annotation, is_typed, nullable
from decikit._engine.wiring.plan import Branch, Call, Loop, Plan, Resolved, ScatterGather, Sequence, NodeVersion
from decikit.ops.nested import struct_item
from decikit._engine.types import columnar_item, decimal_boxer, decimal_scale, item_schema, to_raw
from decikit.ops.text import Text

Values = dict[int, np.ndarray]


class Kernel:
    """Consecutive compiled calls run as one numba kernel over `n` rows.

    `run(values, valid, bundles, n)` reads its input versions from `values`
    (and OPTIONAL inputs' masks from `valid`; absent means all valid), takes
    each call's validated params bundle from `bundles[call.id]` (nodes
    without params need none), and stores the versions it keeps in `values`.
    Values only used inside the kernel are never stored. A version declared
    `T | None` also stores its validity mask in `valid`. If numba can't
    compile the calls together, they run one by one in Python from then on,
    as one `Fallback` each with the reason `"split at launch: <error>"`;
    `on_split`, when set, is called with those fallbacks first and may raise
    to refuse the split.

    Example::

        units = compile_plan(plan)
        unit = units[plan.calls[0].id]
        unit.run(values, {}, bundles, n)
    """

    __slots__ = ("calls", "fn", "reads", "optional", "ragged", "sinks", "writes", "declared", "on_split",
                 "_masked", "_layout", "_tables", "_python", "_trace", "_trace_fn")
    # Whether a kernel numba can't build runs its calls one by one instead.
    splits = True

    def __init__(self, calls, fn, reads, optional, writes, masked, layout, choices, ragged=(), sinks=(),
                 trace=None, declared=()):
        self.calls: tuple[Call, ...] = calls
        # Per call, whether `@allow_fallback` accepts it running in Python.
        self.declared: tuple[bool, ...] = declared or (False,) * len(calls)
        self.on_split = None
        self.fn = fn
        self.reads: tuple[NodeVersion, ...] = reads
        self.optional: tuple[NodeVersion, ...] = optional
        self.ragged: tuple[NodeVersion, ...] = ragged
        self.sinks: tuple[tuple[NodeVersion, Schema], ...] = sinks
        self.writes: tuple[tuple[NodeVersion, np.dtype], ...] = writes
        self._masked: tuple[int, ...] = masked
        self._layout = layout
        self._tables = tuple(None if c is None else CodeTable(*c) for c in choices)
        self._python: tuple[Fallback, ...] = ()
        # `(program, outputs, variables, refs, factor, exact)` for the trace kernel, or None.
        self._trace = trace
        self._trace_fn = None

    def _trace_kernel(self):
        if self._trace_fn is None:
            program, outputs, variables, refs, *_ = self._trace
            self._trace_fn = fused_kernel(program, outputs, variables, trace_refs=refs)
        return self._trace_fn

    def run(self, values: Values, valid: Values, bundles: Mapping[int, tuple], n: int,
            trace=None, table=None) -> None:
        if self._python:
            for fallback in self._python:
                fallback.run(values, valid, bundles, n)
            return
        cols = tuple([values[v.id] for v in self.reads])
        # One shared all-valid mask: a tree can read dozens of nullable columns, and one allocation each showed in a record's run.
        ones = np.ones(n, np.bool_) if self.optional else None
        valids = tuple([valid.get(v.id, ones) for v in self.optional])
        params: list[Any] = []
        for cid, has_params, consts, row in self._layout:
            bundle = bundles[cid] if has_params else ()
            if row:
                params += (bundle, consts)
            else:
                params += bundle
                params += consts
        flats = tuple([a for v in self.ragged for a in values[v.id].arrays])
        sinks = tuple([Sink(n, len(schema)) for _, schema in self.sinks])
        outs = [np.empty(n, dtype) for _, dtype in self.writes]
        masks = [np.empty(n, np.bool_) for _ in self._masked]
        try:
            if trace is None:
                self.fn(n, cols, valids, tuple(params), flats, sinks, tuple(outs + masks))
            else:
                program, outputs, variables, refs, factor, exact = self._trace
                fn = self._trace_kernel()
                header = np.empty(n * len(refs) * factor, np.int64)
                cursor = np.zeros(1, np.int64)
                offsets = np.empty(n, np.int64)
                fn(n, cols, valids, tuple(params), flats, sinks, tuple(outs + masks), header, cursor, offsets)
                trace.drain_kernel(header, offsets, table, int(cursor[0]),
                                   None if not exact else n * len(refs))
        except FALLBACK_ERRORS as e:
            if not self.splits:
                raise
            # Each call compiled alone, but not together (a value numba can't
            # type only reaches it here), so run them one by one in Python.
            reason = f"split at launch: {type(e).__name__}: {e}"
            python = tuple(Fallback(c, getattr(c.node.fn, "py_func", c.node.fn), reason, d)
                           for c, d in zip(self.calls, self.declared))
            if self.on_split is not None:
                # Before the split: a refusal leaves the kernel whole, so the next launch is refused too.
                self.on_split(python)
            self._python = python
            return self.run(values, valid, bundles, n)
        for (v, _), out, table in zip(self.writes, outs, self._tables):
            values[v.id] = out if table is None else coded(out, table, valid, v.id)
        for (v, schema), sink in zip(self.sinks, sinks):
            # `values[v.id]` currently holds the cumulative-length column the wrapper just wrote;
            # turn it into the per-row `[lo, hi)` slices the sink's flat buffer actually holds.
            values[v.id] = _ragged_values(values[v.id], sink, schema)
        for k, mask in zip(self._masked, masks):
            valid[self.writes[k][0].id] = mask


class Fallback:
    """One call that runs on its own, row by row; `reason` says why and `declared` whether the author asked.

    Same `run` and `writes` as `Kernel`; it stores every version it writes.
    """

    __slots__ = ("calls", "fn", "reason", "declared", "writes", "_boxes", "_scales", "_tables", "_whole", "_checks",
                 "_ks", "_dicts", "_python")

    def __init__(self, call: Call, fn, reason: str, declared: bool = False):
        self.calls = (call,)
        self.fn = fn
        self.reason = reason
        self.declared = declared
        # Python values come back as they are, as objects; only a row node returns a coded output as its index.
        self.writes = tuple(
            (v, output_dtype(o.annotation) if (call.node.kind == "row" and coded_values(o.annotation) is not None)
             or is_typed(base_annotation(o.annotation)) else np.dtype(object))
            for v, o in zip(call.writes, call.node.outputs)
        )
        # A Decimal reaches the Python body as a Decimal, and what it returns is stored as its int64.
        node = call.node
        self._boxes = ({i.arg: b for i in node.inputs if (b := decimal_boxer(i.annotation))}
                       | {d.arg: b for d in node.params if (b := decimal_boxer(d.annotation))})
        self._scales = tuple(decimal_scale(o.annotation) for o in node.outputs)
        self._tables = tuple(CodeTable(*c) if node.kind == "row" and (c := coded_values(o.annotation)) is not None
                             else None for o in node.outputs)
        self._ks = range(len(self.writes))
        self._dicts = tuple(to_dicts(o.annotation) for o in node.outputs)
        self._python = not isinstance(fn, Dispatcher)
        self._whole = tuple(dtype == np.int64 for _, dtype in self.writes)
        # A number, Decimal or bool has its declared type already; only an object output can be a wrong shape.
        self._checks = tuple(node.kind == "scalar" and dtype == object for _, dtype in self.writes)

    def run(self, values: Values, valid: Values, bundles: Mapping[int, tuple], n: int) -> None:
        call = self.calls[0]
        node = call.node
        bundle = bundles[call.id] if node.params else ()
        cols = [(i.arg, _per_row(values[v.id], self._python),
                 valid.get(v.id) if i.null_policy is NullPolicy.OPTIONAL else None) for i, v in zip(node.inputs, call.reads)]
        outs = [np.empty(n, dtype) for _, dtype in self.writes]
        masks: list = [None] * len(outs)  # made when a null turns up
        consts = tuple(v for _, v in node.consts)
        fixed = dict(node.consts) | {d.arg: b for d, b in zip(node.params, bundle)}
        boxes = self._boxes
        fixed = {a: boxes[a](x) if a in boxes else x for a, x in fixed.items()}
        scales, whole, ks, dicts = self._scales, self._whole, self._ks, self._dicts
        single = len(outs) == 1 and node.kind == "scalar"
        for r in range(n):
            args = [(a, None if m is not None and not m[r] else boxes[a](col[r]) if a in boxes else col[r])
                    for a, col, m in cols]
            if node.kind == "row":
                result = self.fn(tuple(x for _, x in args), bundle, consts)
            else:
                result = self.fn(**dict(args), **fixed)
            if single:
                result = (result,)
            for k, x in zip(ks, result):
                x = x.text if isinstance(x, Text) else x
                if x is None:
                    if masks[k] is None:
                        masks[k] = np.ones(n, np.bool_)
                    masks[k][r], x = False, 0
                elif scales[k] is not None:
                    x = to_raw(x, scales[k])
                elif dicts[k] is not None:
                    x = dicts[k](x)
                elif whole[k] and isinstance(x, (float, np.floating)) and not x.is_integer():
                    raise refuse(f"step '{node.origin.path}' output '{call.writes[k].name}'"
                                 + (f" (row {r})" if n > 1 else ""), x)
                outs[k][r] = x
        for k, checked in enumerate(self._checks):
            if checked:
                mask = masks[k]
                check_output(node.origin.path, node.outputs[k],
                             outs[k].tolist() if mask is None else [x if m else None for x, m in zip(outs[k], mask)])
        for v, out, mask, table in zip(call.writes, outs, masks, self._tables):
            values[v.id] = out
            if mask is not None:
                valid[v.id] = mask
            if table is not None:
                if mask is not None:
                    out[~mask] = -1
                values[v.id] = coded(out, table, valid, v.id)


def _ragged_values(hi: np.ndarray, sink: Sink, schema: Schema) -> np.ndarray:
    """One `list[dict]` per row, sliced `[lo, hi)` out of `sink`'s flat buffer.

    `hi[r]` is the cumulative item count after row `r` (an ordinary kernel
    output); `lo[r]` is the previous row's `hi`, 0 for row 0. Matches the
    shape a Python-fallback `Columnar[Item]` output already returns, so
    `state.py` needs no separate path for a compiled one.
    """
    lo = np.zeros(len(hi), np.int64)
    lo[1:] = hi[:-1]
    out = np.empty(len(hi), object)
    for r in range(len(hi)):
        out[r] = [{name: (bool(v) if t is bool else int(v) if t is int else float(v))
                   for (name, t), v in zip(schema, item)}
                  for item in sink.data[lo[r]:hi[r]]]
    return out


def _per_row(values: np.ndarray, python: bool) -> list:
    # A kernel that fails at run time retries by calling each step once per row; a `Ragged` (the
    # shared-kernel shape) has no per-row representation of its own, so give it `build_rows`'s.
    if isinstance(values, Ragged):
        return [values.row(i) for i in range(len(values.lo))]
    # A record array's tolist() gives tuples: a compiled body takes each record, a Python body the dict interpreted
    # mode gives it.
    if (names := values.dtype.names) is not None:
        return [dict(zip(names, row)) for row in values.tolist()] if python else list(values)
    return values.tolist()


Unit = Union[Kernel, Fallback]


def compile_plan(plan: Plan, *, fuse: bool = True, python: Mapping[int, str] = {}) -> dict[int, Unit]:
    """Compile a plan's scalar and row calls; returns call id -> the unit that runs it.

    With `fuse=True`, the consecutive scalar calls directly inside one
    sequence share one kernel and a unit keeps only the versions read outside
    it or output by the plan. With `fuse=False` every call is its own unit
    and keeps everything it writes. A row call is always its own unit. A call
    numba can't compile, or one `python` names (call id -> reason), becomes a
    `Fallback`, splitting its kernel around it.
    Frame calls get no unit; branches and loops aren't compiled as a whole,
    only the calls inside them. A unit runs when a walk over `plan.root`
    reaches its first call, `unit.calls[0]`.

    Example::

        units = compile_plan(resolve(pipeline))
        for call in plan.calls:
            unit = units.get(call.id)
            if unit is not None and unit.calls[0] is call:
                unit.run(values, valid, bundles, n)
    """
    keep = _kept(plan) if fuse else None
    units: dict[int, Unit] = {}
    for run in (part for whole in _runs(plan.root, fuse) for part in _split_at_nulls(whole)):
        compiled = [(call, None, getattr(call.node.fn, "py_func", call.node.fn), python[call.id], False)
                    if call.id in python else (call, *compile_call(call.node)) for call in run]
        start = 0
        for k, (call, _, fn, reason, declared) in enumerate(compiled + [(None, None, None, "end", False)]):
            if reason is None:
                continue
            if start < k:
                unit = _kernel(compiled[start:k], keep)
                units.update((c.id, unit) for c in unit.calls)
            if call is not None:
                units[call.id] = Fallback(call, fn, reason, declared)
            start = k + 1
    return units


def _runs(r: Resolved, fuse: bool) -> Iterator[list[Call]]:
    if isinstance(r, Call):
        if r.node.kind != "frame":
            yield [r]
    elif isinstance(r, Sequence):
        run: list[Call] = []
        for child in r.children:
            if fuse and isinstance(child, Call) and child.node.kind == "scalar":
                run.append(child)
                continue
            if run:
                yield run
                run = []
            yield from _runs(child, fuse)
        if run:
            yield run
    elif isinstance(r, Branch):
        yield from _runs(r.condition, fuse)
        for arm in r.arms:
            yield from _runs(arm, fuse)
    elif isinstance(r, Loop):
        yield from _runs(r.condition, fuse)
        yield from _runs(r.body, fuse)
    else:
        yield from _runs(r.body, fuse)
        if r.accumulate is not None:
            yield from _runs(r.accumulate, fuse)


def _split_at_nulls(run: list[Call]) -> Iterator[list[Call]]:
    # A value that may be null reaches its readers through the driver, which
    # applies each reader's null policy, so no kernel both writes and reads one.
    # A struct output is a record scalar (a pointer to a fresh buffer) in flight
    # but a record array in the state, so a reader splits into the next kernel
    # and reads the record array its struct input expects. A ragged (Columnar[Item])
    # output has no in-kernel value at all -- what the kernel produces is the
    # wrapper's item count, not the rows -- so a reader must read the materialised
    # version from `values`, never the count a same-kernel `RES` source would give it.
    part: list[Call] = []
    made: set[int] = set()
    for call in run:
        if any(v.id in made for v in call.reads):
            yield part
            part, made = [], set()
        part.append(call)
        made |= {v.id for v, o in zip(call.writes, call.node.outputs)
                 if nullable(o.annotation) or struct_item(o.annotation) is not None
                 or columnar_item(base_annotation(o.annotation)) is not None}
    if part:
        yield part


def _kept(plan: Plan) -> tuple[set[int], dict[int, set[int]]] | None:
    # A frame call of unknown inputs reads whatever is in the frame.
    # ponytail: keeps every version for such a plan; narrow to the latest version per name if it matters.
    if any(c.reads is None for c in plan.calls):
        return None
    pinned = {v.id for v in plan.outputs.values()}
    stack: list[Resolved] = [plan.root]
    while stack:
        r = stack.pop()
        if isinstance(r, Sequence):
            stack += r.children
        elif isinstance(r, Branch):
            pinned |= {v.id for m in r.merges for v in (m.version, m.prior, *m.arms) if v is not None}
            pinned.add(r.condition.writes[0].id)
            stack += r.arms
        elif isinstance(r, Loop):
            pinned |= {v.id for c in r.carries for v in (c.version, c.initial, c.last)}
            pinned.add(r.condition.writes[0].id)
            stack.append(r.body)
        elif isinstance(r, ScatterGather):
            pinned |= {v.id for v in r.new_fields}
            pinned.add(r.column.id)
            pinned.add(r.out.id)
            stack.append(r.body)
            if r.accumulate is not None:
                stack.append(r.accumulate)
    readers: dict[int, set[int]] = {}
    for c in plan.calls:
        for v in c.reads:
            readers.setdefault(v.id, set()).add(c.id)
    return pinned, readers


def _decimal(source: tuple, annotation: Any) -> tuple:
    scale = decimal_scale(annotation)
    return source if scale is None else (SrcKind.DEC, scale, source)


class Layout:
    """Where each value of one kernel comes from: the columns it reads, the params it takes, the calls' results.

    `spec(call, key, fn)` numbers the calls in the order it sees them;
    `produced` maps a version made inside the kernel to its source.
    """

    def __init__(self) -> None:
        self.reads: list[NodeVersion] = []
        self.optional: list[NodeVersion] = []
        self.ragged: list[NodeVersion] = []
        self.sinks: list[tuple[NodeVersion, Schema]] = []
        self.layout: list[tuple] = []
        self.produced: dict[int, tuple] = {}
        self._cols: dict[int, int] = {}
        self._masks: dict[int, int] = {}
        self._rag: dict[int, tuple] = {}
        self._flats = 0
        self._p = 0
        self._specs = 0
        self._sinks = 0

    def source(self, v: NodeVersion, inp: Input | None = None) -> tuple:
        if v.id in self.produced:
            return self.produced[v.id]
        if inp is not None and (item := columnar_item(inp.annotation)) is not None:
            schema = item_schema(item)
            source = self._rag.get(v.id)
            if source is None:
                source = self._rag[v.id] = (SrcKind.RAG, schema, self._flats)
                self._flats += 2 + len(schema)
                self.ragged.append(v)
            elif source[1] != schema:
                # One `flats` block per column, so a second Item would silently get the first's fields.
                raise TypeError(f"'{v.name}' is read as Columnar[...] with two different Item types in one "
                                "kernel; read it with one Item type")
            return source
        if v.id not in self._cols:
            self._cols[v.id] = len(self.reads)
            self.reads.append(v)
        # A null `bytes` value is a span of length -1, so it needs no mask.
        if inp is None or inp.null_policy is not NullPolicy.OPTIONAL or base_annotation(inp.annotation) is bytes:
            return (SrcKind.COL, self._cols[v.id])
        if v.id not in self._masks:
            self._masks[v.id] = len(self.optional)
            self.optional.append(v)
        return (SrcKind.OPT, self._cols[v.id], self._masks[v.id])

    def spec(self, call: Call, key: str, fn) -> Spec:
        node = call.node
        sources = [self.source(v, inp) for inp, v in zip(node.inputs, call.reads)]
        if node.kind != "row":
            sources = [_decimal(src, inp.annotation) for src, inp in zip(sources, node.inputs)]
        consts = tuple(v for _, v in node.consts)
        if node.params or consts or node.kind == "row":
            self.layout.append((call.id, bool(node.params), consts, node.kind == "row"))
        p = self._p
        ragged_item = columnar_item(base_annotation(node.outputs[0].annotation)) if len(node.outputs) == 1 else None
        if node.kind == "row":
            args = ((SrcKind.ROW, tuple(sources)), (SrcKind.PAR, p), (SrcKind.PAR, p + 1))
            self._p += 2
        else:
            by_arg = {i.arg: src for i, src in zip(node.inputs, sources)}
            by_arg |= {d.arg: _decimal((SrcKind.PAR, p + k), d.annotation) for k, d in enumerate(node.params)}
            p += len(node.params)
            by_arg |= {name: (SrcKind.PAR, p + k) for k, (name, _) in enumerate(node.consts)}
            self._p = p + len(node.consts)
            # `node.fn`'s own names, not `fn`'s: a ragged output's `fn` is the wrapper, whose own
            # signature is `(packed, sink)` and says nothing about the step's real arguments; every
            # other `fn` agrees with `node.fn` on names already, since only its globals are rewritten.
            names = parameters(node.fn)
            missing = [a for a in names if a not in by_arg]
            if missing:
                raise ValueError(f"{node.origin.path}: argument(s) {missing} are not an input, const or param")
            if ragged_item is not None:
                # The wrapper's own signature is fixed arity (packed args, sink), not `*args`: a
                # star-args callee has no per-argument type kernel.py's hand-rolled IR can bind to.
                args = ((SrcKind.ROW, tuple(by_arg[a] for a in names)), (SrcKind.SINK, self._sinks))
                self.sinks.append((call.writes[0], item_schema(ragged_item)))
                self._sinks += 1
            else:
                args = tuple(by_arg[a] for a in names)
        dtypes = tuple(output_dtype(o.annotation) for o in node.outputs)
        nulls = tuple(nullable(o.annotation) for o in node.outputs)
        structs = tuple(struct_item(base_annotation(o.annotation)) is not None for o in node.outputs)
        scales = tuple(decimal_scale(o.annotation) for o in node.outputs)
        s, self._specs = self._specs, self._specs + 1
        self.produced.update((v.id, (SrcKind.RES, s, k)) for k, v in enumerate(call.writes))
        return Spec(key, fn, args, dtypes, node.kind == "row" or len(dtypes) > 1,
                    nulls if any(nulls) else (), structs if any(structs) else (),
                    scales if any(sc is not None for sc in scales) else ())


def _kernel(compiled: list, keep) -> Kernel:
    calls = tuple(c for c, *_ in compiled)
    ids = {c.id for c in calls}
    lay = Layout()
    specs, outputs, writes, masked, choices = [], [], [], [], []
    for call, key, fn, *_ in compiled:
        spec = lay.spec(call, key, fn)
        specs.append(spec)
        for k, v in enumerate(call.writes):
            # A ragged output's write always lands in `values[v.id]`: `Kernel.run` reads it back
            # (the wrapper's hi-count) to materialise the sink into the version's real rows,
            # whether or not anything else in the plan reads that version.
            ragged = columnar_item(base_annotation(call.node.outputs[k].annotation)) is not None
            if ragged or keep is None or v.id in keep[0] or keep[1].get(v.id, set()) - ids:
                if spec.nullable and spec.nullable[k]:
                    masked.append(len(writes))
                outputs.append(lay.produced[v.id])
                writes.append((v, spec.dtypes[k]))
                choices.append(coded_values(call.node.outputs[k].annotation))
    fn = fused_kernel(tuple(specs), tuple(outputs))
    # A straight-line kernel runs every call once per row, so its trace emits
    # exactly `n * len(calls)` STEP events; refs are the plan-global call refs.
    trace = (tuple(specs), tuple(outputs), (), tuple(c.id + 1 for c in calls), 1, True)
    return Kernel(calls, fn, tuple(lay.reads), tuple(lay.optional), tuple(writes), tuple(masked), tuple(lay.layout),
                  tuple(choices), tuple(lay.ragged), tuple(lay.sinks), trace, tuple(d for *_, d in compiled))

