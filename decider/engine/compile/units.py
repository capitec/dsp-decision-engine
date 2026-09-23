from __future__ import annotations

import types
import typing
from typing import Any, Iterator, Mapping, Union

import numpy as np

from decider.engine.compile.kernel import Spec, fused_kernel
from decider.engine.compile.njit import compile_call, numpy_dtype, parameters
from decider.engine.ir.decls import NullPolicy, base_annotation
from decider.engine.ir.origin import Origin
from decider.engine.wiring.plan import Branch, Call, Loop, Plan, Resolved, Sequence, Version

Values = dict[int, np.ndarray]


class Kernel:
    """Consecutive compiled calls run as one numba kernel over `n` rows.

    `run(values, valid, bundles, n)` reads its input versions from `values`
    (and OPTIONAL inputs' masks from `valid`; absent means all valid), takes
    each call's validated params bundle from `bundles[call.id]` (nodes
    without params need none), and stores the versions it keeps in `values`.
    Values only used inside the kernel are never stored. A version declared
    `T | None` also stores its validity mask in `valid`.

    Example::

        units = compile_plan(plan)
        unit = units[plan.calls[0].id]
        unit.run(values, {}, bundles, n)
        [o.path for o in unit.origins]
    """

    __slots__ = ("calls", "fn", "reads", "optional", "writes", "_masked", "_layout")

    def __init__(self, calls, fn, reads, optional, writes, masked, layout):
        self.calls: tuple[Call, ...] = calls
        self.fn = fn
        self.reads: tuple[Version, ...] = reads
        self.optional: tuple[Version, ...] = optional
        self.writes: tuple[tuple[Version, np.dtype], ...] = writes
        self._masked: tuple[int, ...] = masked
        self._layout = layout

    @property
    def origins(self) -> tuple[Origin, ...]:
        return tuple(c.node.origin for c in self.calls)

    def run(self, values: Values, valid: Values, bundles: Mapping[int, tuple], n: int) -> None:
        cols = tuple([values[v.id] for v in self.reads])
        valids = tuple([valid[v.id] if v.id in valid else np.ones(n, np.bool_) for v in self.optional])
        params: list[Any] = []
        for cid, has_params, consts, row in self._layout:
            bundle = bundles[cid] if has_params else ()
            if row:
                params += (bundle, consts)
            else:
                params += bundle
                params += consts
        outs = [np.empty(n, dtype) for _, dtype in self.writes]
        masks = [np.empty(n, np.bool_) for _ in self._masked]
        self.fn(n, cols, valids, tuple(params), tuple(outs + masks))
        for (v, _), out in zip(self.writes, outs):
            values[v.id] = out
        for k, mask in zip(self._masked, masks):
            valid[self.writes[k][0].id] = mask


class Fallback:
    """One call numba couldn't compile, run row by row in Python; `reason` says why.

    Same `run`, `origins` and `writes` as `Kernel`; it stores every version it writes.
    """

    __slots__ = ("calls", "fn", "reason", "writes")

    def __init__(self, call: Call, fn, reason: str):
        self.calls = (call,)
        self.fn = fn
        self.reason = reason
        self.writes = tuple((v, output_dtype(o.annotation)) for v, o in zip(call.writes, call.node.outputs))

    @property
    def origins(self) -> tuple[Origin, ...]:
        return (self.calls[0].node.origin,)

    def run(self, values: Values, valid: Values, bundles: Mapping[int, tuple], n: int) -> None:
        call = self.calls[0]
        node = call.node
        bundle = bundles[call.id] if node.params else ()
        cols = [(i.arg, values[v.id], valid.get(v.id) if i.null_policy is NullPolicy.OPTIONAL else None)
                for i, v in zip(node.inputs, call.reads)]
        outs = [np.empty(n, dtype) for _, dtype in self.writes]
        masks = [np.ones(n, np.bool_) for _ in outs]
        consts = tuple(v for _, v in node.consts)
        fixed = dict(node.consts) | {d.arg: b for d, b in zip(node.params, bundle)}
        for r in range(n):
            args = [(a, None if m is not None and not m[r] else col[r]) for a, col, m in cols]
            if node.kind == "row":
                result = self.fn(tuple(x for _, x in args), bundle, consts)
            else:
                result = self.fn(**dict(args), **fixed)
            if len(outs) == 1 and node.kind == "scalar":
                result = (result,)
            for out, mask, x in zip(outs, masks, result):
                if x is None:
                    mask[r], x = False, 0
                out[r] = x
        for v, out, mask in zip(call.writes, outs, masks):
            values[v.id] = out
            if not mask.all():
                valid[v.id] = mask


Unit = Union[Kernel, Fallback]


def compile_plan(plan: Plan, *, fuse: bool = True) -> dict[int, Unit]:
    """Compile a plan's scalar and row calls; returns call id -> the unit that runs it.

    With `fuse=True`, the consecutive scalar calls directly inside one
    sequence share one kernel and a unit keeps only the versions read outside
    it or output by the plan. With `fuse=False` every call is its own unit
    and keeps everything it writes. A row call is always its own unit. A call
    numba can't compile becomes a `Fallback`, splitting its kernel around it.
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
        compiled = [(call, *compile_call(call.node)) for call in run]
        start = 0
        for k, (call, _, fn, reason) in enumerate(compiled + [(None, None, None, "end")]):
            if reason is None:
                continue
            if start < k:
                unit = _kernel(compiled[start:k], keep)
                units.update((c.id, unit) for c in unit.calls)
            if call is not None:
                units[call.id] = Fallback(call, fn, reason)
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
    else:
        yield from _runs(r.condition, fuse)
        yield from _runs(r.body, fuse)


def _split_at_nulls(run: list[Call]) -> Iterator[list[Call]]:
    # A value that may be null reaches its readers through the driver, which
    # applies each reader's null policy, so no kernel both writes and reads one.
    part: list[Call] = []
    made: set[int] = set()
    for call in run:
        if any(v.id in made for v in call.reads):
            yield part
            part, made = [], set()
        part.append(call)
        made |= {v.id for v, o in zip(call.writes, call.node.outputs) if nullable(o.annotation)}
    if part:
        yield part


def nullable(annotation: Any) -> bool:
    """Whether an output declared `annotation` may be `None` (`T | None`).

    >>> nullable(float | None), nullable(float)
    (True, False)
    """
    return typing.get_origin(annotation) in (typing.Union, types.UnionType) and type(None) in typing.get_args(annotation)


def output_dtype(annotation: Any) -> np.dtype:
    """The dtype a kernel stores an output declared `annotation` in: `numpy_dtype` of `T` for `T | None`.

    >>> output_dtype(int | None)
    dtype('int64')
    """
    return numpy_dtype(base_annotation(annotation))


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
    readers: dict[int, set[int]] = {}
    for c in plan.calls:
        for v in c.reads:
            readers.setdefault(v.id, set()).add(c.id)
    return pinned, readers


def _kernel(compiled: list, keep) -> Kernel:
    calls = tuple(c for c, *_ in compiled)
    ids = {c.id for c in calls}
    cols: dict[int, int] = {}
    masks: dict[int, int] = {}
    reads: list[Version] = []
    optional: list[Version] = []
    produced: dict[int, tuple[int, int]] = {}
    specs, layout, outputs, writes, masked = [], [], [], [], []
    p = 0
    for s, (call, key, fn, _) in enumerate(compiled):
        node = call.node
        sources = []
        for inp, v in zip(node.inputs, call.reads):
            if v.id in produced:
                sources.append(("res", *produced[v.id]))
                continue
            if v.id not in cols:
                cols[v.id] = len(reads)
                reads.append(v)
            if inp.null_policy is NullPolicy.OPTIONAL:
                if v.id not in masks:
                    masks[v.id] = len(optional)
                    optional.append(v)
                sources.append(("opt", cols[v.id], masks[v.id]))
            else:
                sources.append(("col", cols[v.id]))
        consts = tuple(v for _, v in node.consts)
        if node.params or consts or node.kind == "row":
            layout.append((call.id, bool(node.params), consts, node.kind == "row"))
        if node.kind == "row":
            args = (("row", tuple(sources)), ("par", p), ("par", p + 1))
            p += 2
        else:
            by_arg = {i.arg: src for i, src in zip(node.inputs, sources)}
            by_arg |= {d.arg: ("par", p + k) for k, d in enumerate(node.params)}
            p += len(node.params)
            by_arg |= {name: ("par", p + k) for k, (name, _) in enumerate(node.consts)}
            p += len(node.consts)
            missing = [a for a in parameters(fn) if a not in by_arg]
            if missing:
                raise ValueError(f"{node.origin.path}: argument(s) {missing} are not an input, const or param")
            args = tuple(by_arg[a] for a in parameters(fn))
        dtypes = tuple(output_dtype(o.annotation) for o in node.outputs)
        nulls = tuple(nullable(o.annotation) for o in node.outputs)
        specs.append(Spec(key, fn, args, dtypes, node.kind == "row" or len(dtypes) > 1, nulls if any(nulls) else ()))
        for k, v in enumerate(call.writes):
            produced[v.id] = (s, k)
            if keep is None or v.id in keep[0] or keep[1].get(v.id, set()) - ids:
                if nulls[k]:
                    masked.append(len(writes))
                outputs.append((s, k))
                writes.append((v, dtypes[k]))
    fn = fused_kernel(tuple(specs), tuple(outputs))
    return Kernel(calls, fn, tuple(reads), tuple(optional), tuple(writes), tuple(masked), tuple(layout))
