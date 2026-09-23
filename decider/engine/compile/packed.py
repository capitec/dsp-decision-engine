from __future__ import annotations

from typing import Iterator

from decider.engine.compile.kernel import Fork, Repeat, fused_kernel
from decider.engine.compile.njit import compile_call
from decider.engine.compile.units import Kernel, Layout, output_dtype
from decider.engine.ir.decls import TYPED, NullPolicy, base_annotation, nullable
from decider.engine.wiring.plan import Branch, Call, Loop, Plan, Resolved, Sequence, Version


class Packed(Kernel):
    """A whole branch or loop as one kernel: each row takes its own arm, or its own iterations, inside it.

    Runs like a `Kernel` and stores only what the branch merges or the loop
    carries. `inner` holds every version made inside it; `passthrough` the
    versions it copies without a call reading them (a merge's earlier
    value, a carry's initial one); `guarded` the versions that, with a null
    on a row it runs, must take the step-by-step path instead, where a step
    no row reaches can't fail on one. If numba can't build it, `run` raises
    rather than running its calls one by one.

    Example::

        packed = compile_packed(plan)["by_sector"]
        packed.run(values, {}, bundles, n)
    """

    __slots__ = ("inner", "passthrough", "guarded")
    splits = False


class _Unpackable(Exception):
    pass


def compile_packed(plan: Plan, lazy: bool = False) -> dict[str, Packed]:
    """Every branch and loop that runs as one kernel, by path.

    One packs when every call in it (condition, arms, body, nested branches
    and loops) is a scalar call numba compiles, with no nullable output;
    every merged or carried name is a `float`, `int` or `bool` of one dtype
    throughout; and nothing made inside it but what it merges or carries is
    a pipeline output. With `lazy` params validation, a call that may not
    run (anything but its own condition) must also have no params, since
    launching the kernel validates every call in it.

    Example::

        packed = compile_packed(resolve(pipeline))
        sorted(packed)   # ["by_sector"]
    """
    outputs = {v.id for v in plan.outputs.values()}
    found = {}
    for r in _constructs(plan.root):
        try:
            found[r.node.origin.path] = _pack(r, outputs, lazy)
        except _Unpackable:
            pass
    return found


def _constructs(r: Resolved) -> Iterator[Branch | Loop]:
    if isinstance(r, Sequence):
        for child in r.children:
            yield from _constructs(child)
    elif isinstance(r, Branch):
        yield r
        for arm in r.arms:
            yield from _constructs(arm)
    elif isinstance(r, Loop):
        yield r
        yield from _constructs(r.body)


def _calls(r: Resolved) -> Iterator[Call]:
    if isinstance(r, Call):
        yield r
    elif isinstance(r, Sequence):
        for child in r.children:
            yield from _calls(child)
    elif isinstance(r, Branch):
        yield r.condition
        for arm in r.arms:
            yield from _calls(arm)
    else:
        yield r.condition
        yield from _calls(r.body)


def _pack(top: Branch | Loop, outputs: set[int], lazy: bool) -> Packed:
    calls = list(_calls(top))
    conditional = calls[1:]
    if lazy and any(c.node.params for c in conditional):
        raise _Unpackable
    compiled = {}
    for c in calls:
        if c.node.kind != "scalar" or any(nullable(o.annotation) for o in c.node.outputs):
            raise _Unpackable
        key, fn, reason = compile_call(c.node)
        if reason is not None:
            raise _Unpackable
        compiled[c.id] = (key, fn)
    lay = Layout()
    variables: list = []
    inner = {v.id for c in calls for v in c.writes}
    passthrough: list[Version] = []

    def var(v: Version) -> int:
        if base_annotation(v.annotation) not in TYPED:
            raise _Unpackable
        inner.add(v.id)
        lay.produced[v.id] = ("var", len(variables))
        variables.append(output_dtype(v.annotation))
        return len(variables) - 1

    def value(v: Version, into: int) -> tuple:
        if output_dtype(v.annotation) != variables[into]:
            raise _Unpackable
        if v.id not in inner and v not in passthrough:
            passthrough.append(v)
        return lay.source(v)

    def block(r: Resolved) -> list:
        if isinstance(r, Call):
            return [lay.spec(r, *compiled[r.id])]
        if isinstance(r, Sequence):
            return [x for child in r.children for x in block(child)]
        if isinstance(r, Branch):
            program = block(r.condition)
            test = lay.produced[r.condition.writes[0].id]
            arms = tuple(tuple(block(arm)) for arm in r.arms)
            slots = [var(m.version) for m in r.merges]
            merges = tuple((s, tuple(value(m.arms[k] or m.prior, s) for k in range(len(arms))))
                           for s, m in zip(slots, r.merges))
            return program + [Fork(test, arms, merges)]
        slots = [var(c.version) for c in r.carries]
        carries = tuple((s, value(c.initial, s)) for s, c in zip(slots, r.carries))
        condition = tuple(block(r.condition))
        test = lay.produced[r.condition.writes[0].id]
        body = tuple(block(r.body))
        updates = tuple((s, value(c.last, s)) for s, c in zip(slots, r.carries))
        return [Repeat(carries, condition, test, body, updates, r.node.max_iterations)]

    program = tuple(block(top))
    kept = [m.version for m in top.merges] if isinstance(top, Branch) else [c.version for c in top.carries]
    if outputs & (inner - {v.id for v in kept}):
        raise _Unpackable
    writes = tuple((v, variables[lay.produced[v.id][1]]) for v in kept)
    fn = fused_kernel(program, tuple(lay.produced[v.id] for v in kept), tuple(variables))
    # Only float, int and bool are merged or carried, so no output is a Literal code.
    packed = Packed(tuple(calls), fn, tuple(lay.reads), tuple(lay.optional), writes, (), tuple(lay.layout),
                    (None,) * len(writes))
    packed.inner = frozenset(inner)
    packed.passthrough = tuple(passthrough)
    required = {v.id: v for c in conditional for i, v in zip(c.node.inputs, c.reads)
                if v.id not in inner and i.null_policy is NullPolicy.REQUIRED}
    packed.guarded = tuple(passthrough) + tuple(v for v in required.values() if v not in passthrough)
    return packed
