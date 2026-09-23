from __future__ import annotations

import typing
from typing import Any, Iterator, Mapping, Union

import numpy as np

from decider.engine.compile.kernel import Spec, fused_kernel
from decider.engine.compile.njit import FALLBACK_ERRORS, compile_call, numpy_dtype, parameters
from decider.engine.ir.decls import Input, NullPolicy, base_annotation, nullable
from decider.engine.wiring.plan import Branch, Call, Loop, Plan, Resolved, Sequence, Version

Values = dict[int, np.ndarray]


class Kernel:
    """Consecutive compiled calls run as one numba kernel over `n` rows.

    `run(values, valid, bundles, n)` reads its input versions from `values`
    (and OPTIONAL inputs' masks from `valid`; absent means all valid), takes
    each call's validated params bundle from `bundles[call.id]` (nodes
    without params need none), and stores the versions it keeps in `values`.
    Values only used inside the kernel are never stored. A version declared
    `T | None` also stores its validity mask in `valid`. If numba can't
    compile the calls together, they run one by one in Python from then on.

    Example::

        units = compile_plan(plan)
        unit = units[plan.calls[0].id]
        unit.run(values, {}, bundles, n)
    """

    __slots__ = ("calls", "fn", "reads", "optional", "writes", "_masked", "_layout", "_choices", "_python")
    # Whether a kernel numba can't build runs its calls one by one instead.
    splits = True

    def __init__(self, calls, fn, reads, optional, writes, masked, layout, choices):
        self.calls: tuple[Call, ...] = calls
        self.fn = fn
        self.reads: tuple[Version, ...] = reads
        self.optional: tuple[Version, ...] = optional
        self.writes: tuple[tuple[Version, np.dtype], ...] = writes
        self._masked: tuple[int, ...] = masked
        self._layout = layout
        self._choices: tuple[tuple | None, ...] = choices
        self._python: tuple[Fallback, ...] = ()

    def run(self, values: Values, valid: Values, bundles: Mapping[int, tuple], n: int) -> None:
        if self._python:
            for fallback in self._python:
                fallback.run(values, valid, bundles, n)
            return
        cols = tuple([values[v.id] for v in self.reads])
        # One shared all-valid mask: a tree can read dozens of nullable columns, and one allocation each showed in score().
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
        outs = [np.empty(n, dtype) for _, dtype in self.writes]
        masks = [np.empty(n, np.bool_) for _ in self._masked]
        try:
            self.fn(n, cols, valids, tuple(params), tuple(outs + masks))
        except FALLBACK_ERRORS as e:
            if not self.splits:
                raise
            # Each call compiled alone, but not together (a value numba can't
            # type only reaches it here), so run them one by one in Python.
            reason = f"{type(e).__name__}: {e}"
            self._python = tuple(Fallback(c, getattr(c.node.fn, "py_func", c.node.fn), reason) for c in self.calls)
            return self.run(values, valid, bundles, n)
        for (v, _), out, choices in zip(self.writes, outs, self._choices):
            values[v.id] = out if choices is None else _decode(out, choices, valid, v.id)
        for k, mask in zip(self._masked, masks):
            valid[self.writes[k][0].id] = mask


class Fallback:
    """One call numba couldn't compile, run row by row in Python; `reason` says why.

    Same `run` and `writes` as `Kernel`; it stores every version it writes.
    """

    __slots__ = ("calls", "fn", "reason", "writes")

    def __init__(self, call: Call, fn, reason: str):
        self.calls = (call,)
        self.fn = fn
        self.reason = reason
        # Python values come back as they are: a `str` output is stored as an object, not a code.
        self.writes = tuple(
            (v, np.dtype(object) if base_annotation(o.annotation) is str else output_dtype(o.annotation))
            for v, o in zip(call.writes, call.node.outputs)
        )

    def run(self, values: Values, valid: Values, bundles: Mapping[int, tuple], n: int) -> None:
        call = self.calls[0]
        node = call.node
        bundle = bundles[call.id] if node.params else ()
        cols = [(i.arg, values[v.id].tolist(), valid.get(v.id) if i.null_policy is NullPolicy.OPTIONAL else None)
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
        for v, out, mask, o in zip(call.writes, outs, masks, node.outputs):
            values[v.id] = out
            if not mask.all():
                valid[v.id] = mask
            choices = literal_choices(o.annotation)
            if choices is not None:
                values[v.id] = _decode(out, choices, valid, v.id)


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


def output_dtype(annotation: Any) -> np.dtype:
    """The dtype a kernel stores an output declared `annotation` in: `numpy_dtype` of `T` for `T | None`.

    A `Literal` of strings is stored as the int64 index of its value.

    >>> output_dtype(int | None)
    dtype('int64')
    """
    if literal_choices(annotation) is not None:
        return np.dtype(np.int64)
    return numpy_dtype(base_annotation(annotation))


def literal_choices(annotation: Any) -> tuple[str, ...] | None:
    """The values of a `Literal` of strings, else `None`.

    A row node's output declared `Literal["low", "high"]` is returned by its
    compiled `fn` as the index of the value (-1 for null) and stored as the
    string; its `reference` returns the string itself.

    >>> literal_choices(typing.Literal["low", "high"]), literal_choices(str)
    (('low', 'high'), None)
    """
    if typing.get_origin(annotation) is typing.Literal and all(isinstance(a, str) for a in typing.get_args(annotation)):
        return typing.get_args(annotation)
    return None


def _decode(codes: np.ndarray, choices: tuple[str, ...], valid: Values, vid: int) -> np.ndarray:
    strings = np.array((*choices, None), object)[codes]
    if (codes < 0).any():
        valid[vid] = codes >= 0
    return strings


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


class Layout:
    """Where each value of one kernel comes from: the columns it reads, the params it takes, the calls' results.

    `spec(call, key, fn)` numbers the calls in the order it sees them;
    `produced` maps a version made inside the kernel to its source.
    """

    def __init__(self) -> None:
        self.reads: list[Version] = []
        self.optional: list[Version] = []
        self.layout: list[tuple] = []
        self.produced: dict[int, tuple] = {}
        self._cols: dict[int, int] = {}
        self._masks: dict[int, int] = {}
        self._p = 0
        self._specs = 0

    def source(self, v: Version, inp: Input | None = None) -> tuple:
        if v.id in self.produced:
            return self.produced[v.id]
        if v.id not in self._cols:
            self._cols[v.id] = len(self.reads)
            self.reads.append(v)
        # A null `bytes` value is a span of length -1, so it needs no mask.
        if inp is None or inp.null_policy is not NullPolicy.OPTIONAL or base_annotation(inp.annotation) is bytes:
            return ("col", self._cols[v.id])
        if v.id not in self._masks:
            self._masks[v.id] = len(self.optional)
            self.optional.append(v)
        return ("opt", self._cols[v.id], self._masks[v.id])

    def spec(self, call: Call, key: str, fn) -> Spec:
        node = call.node
        sources = [self.source(v, inp) for inp, v in zip(node.inputs, call.reads)]
        consts = tuple(v for _, v in node.consts)
        if node.params or consts or node.kind == "row":
            self.layout.append((call.id, bool(node.params), consts, node.kind == "row"))
        p = self._p
        if node.kind == "row":
            args = (("row", tuple(sources)), ("par", p), ("par", p + 1))
            self._p += 2
        else:
            by_arg = {i.arg: src for i, src in zip(node.inputs, sources)}
            by_arg |= {d.arg: ("par", p + k) for k, d in enumerate(node.params)}
            p += len(node.params)
            by_arg |= {name: ("par", p + k) for k, (name, _) in enumerate(node.consts)}
            self._p = p + len(node.consts)
            missing = [a for a in parameters(fn) if a not in by_arg]
            if missing:
                raise ValueError(f"{node.origin.path}: argument(s) {missing} are not an input, const or param")
            args = tuple(by_arg[a] for a in parameters(fn))
        dtypes = tuple(output_dtype(o.annotation) for o in node.outputs)
        nulls = tuple(nullable(o.annotation) for o in node.outputs)
        s, self._specs = self._specs, self._specs + 1
        self.produced.update((v.id, ("res", s, k)) for k, v in enumerate(call.writes))
        return Spec(key, fn, args, dtypes, node.kind == "row" or len(dtypes) > 1, nulls if any(nulls) else ())


def _kernel(compiled: list, keep) -> Kernel:
    calls = tuple(c for c, *_ in compiled)
    ids = {c.id for c in calls}
    lay = Layout()
    specs, outputs, writes, masked, choices = [], [], [], [], []
    for call, key, fn, _ in compiled:
        spec = lay.spec(call, key, fn)
        specs.append(spec)
        for k, v in enumerate(call.writes):
            if keep is None or v.id in keep[0] or keep[1].get(v.id, set()) - ids:
                if spec.nullable and spec.nullable[k]:
                    masked.append(len(writes))
                outputs.append(lay.produced[v.id])
                writes.append((v, spec.dtypes[k]))
                choices.append(literal_choices(call.node.outputs[k].annotation))
    fn = fused_kernel(tuple(specs), tuple(outputs))
    return Kernel(calls, fn, tuple(lay.reads), tuple(lay.optional), tuple(writes), tuple(masked), tuple(lay.layout),
                  tuple(choices))

