from __future__ import annotations

import operator
from typing import Any, Callable, NamedTuple

import numpy as np
from numba import from_dtype, njit, typeof
from numba.core import cgutils, types
from numba.core.typing import signature
from numba.extending import intrinsic


class Spec(NamedTuple):
    """One call inside a fused kernel.

    `args` holds one source per positional argument of `fn`:
    `("col", j)` is element i of input column j; `("opt", j, m)` is the same
    element, or `None` where validity mask m is false; `("res", s, k)` is
    output k of call s of this kernel; `("var", v)` is variable v (set by a
    `Fork` or `Repeat`); `("par", p)` is `params[p]`;
    `("row", (source, ...))` is a tuple of sources. `dtypes` are the declared
    dtypes of the outputs; `unpack` means `fn` returns a tuple of them.
    `nullable[k]` marks an output declared `T | None`: it may be `None`.
    """

    key: str
    fn: Callable
    args: tuple
    dtypes: tuple[np.dtype, ...]
    unpack: bool
    nullable: tuple[bool, ...] = ()


class Fork(NamedTuple):
    """A branch inside a kernel: runs the arm `test` picks, then sets each merged variable.

    `test` is a source holding a bool (true picks arm 0, false arm 1) or an
    int (the arm's index; one out of range raises `ArmOutOfRange`). `arms`
    holds one program per arm; `merges[m] = (var, sources)` sets variable
    `var` to `sources[k]` after arm k ran.
    """

    test: tuple
    arms: tuple[tuple, ...]
    merges: tuple[tuple[int, tuple], ...]


class Repeat(NamedTuple):
    """A loop inside a kernel.

    Sets each carried variable from its source in `carries`, then up to
    `max_iterations` times: runs `condition`, stops unless `test` is true,
    runs `body` and sets each variable from its source in `updates`.
    """

    carries: tuple[tuple[int, tuple], ...]
    condition: tuple
    test: tuple
    body: tuple
    updates: tuple[tuple[int, tuple], ...]
    max_iterations: int


class ArmOutOfRange(ValueError):
    """An int branch condition inside a kernel picked an arm the branch doesn't have."""


# ponytail: unbounded like the step dispatchers; evict with them if sessions ever churn through edits.
_KERNELS: dict[tuple, Callable] = {}


def fused_kernel(program: tuple, outputs: tuple[tuple, ...], variables: tuple[np.dtype, ...] = ()) -> Callable:
    """One njit kernel `kernel(n, cols, valids, params, outs)` running `program` for each of n rows.

    `program` is a tuple of `Spec` (one call), `Fork` and `Repeat`; calls are
    numbered in the order they appear, depth first, which is the `s` of a
    `("res", s, j)` source. Variable v has dtype `variables[v]`. `outputs[k]`
    is the source written into `outs[k]`: a variable, or a result of a call
    at the top level of `program`. Every value is cast to its declared dtype
    when produced, so a fused value matches the one a single-call kernel
    stores. A nullable output also writes its validity into a bool array,
    one per nullable output in order, after the value arrays in `outs`.
    Kernels are shared by content: the same program gives the same kernel,
    whatever the nodes are called.

    Example::

        kernel = fused_kernel((Spec(key, fn, (("col", 0),), (np.dtype("f8"),), False),), (("res", 0, 0),))
        kernel(n, (x,), (), (), (out,))
    """
    key = (_key(program), outputs, variables)
    kernel = _KERNELS.get(key)
    if kernel is None:
        body = _row_body(program, outputs, variables)

        def run(n, cols, valids, params, outs):
            for i in range(n):
                body(cols, valids, params, outs, i)

        kernel = _KERNELS[key] = njit(nogil=True)(run)
    return kernel


def _key(program: tuple) -> tuple:
    out = []
    for x in program:
        if isinstance(x, Spec):
            out.append((x.key, x.args, x.dtypes, x.unpack, x.nullable))
        elif isinstance(x, Fork):
            out.append(("fork", x.test, tuple(_key(a) for a in x.arms), x.merges))
        else:
            out.append(("repeat", x.carries, _key(x.condition), x.test, _key(x.body), x.updates, x.max_iterations))
    return tuple(out)


def _row_body(program: tuple, outputs: tuple[tuple, ...], variables: tuple[np.dtype, ...]) -> Any:
    # Lowered straight to IR: each call goes through the same
    # get_call_type/get_function/cast sequence numba uses for a direct call in
    # jitted source, and results stay SSA values for the calls after it.
    # Forks and loops are plain LLVM blocks; values crossing them live in
    # stack slots, which LLVM promotes back to registers.
    @intrinsic
    def body(typingctx, cols, valids, params, outs, i):
        def codegen(context, builder, sig, args):
            cols_v, valids_v, params_v, outs_v, i_v = args
            results: list[list[tuple[Any, Any]]] = []
            vtypes = [from_dtype(d) for d in variables]
            slots = [cgutils.alloca_once(builder, context.get_value_type(t)) for t in vtypes]

            def element(tup, tup_v, j):
                arr = tup.types[j]
                if arr.ndim == 2:
                    # A `bytes` column: row i is its `(address, byte length)` span.
                    span = types.UniTuple(types.int64, 2)
                    getitem = context.get_function(operator.getitem, signature(arr.dtype, arr, types.UniTuple(i, 2)))
                    parts = [getitem(builder, (builder.extract_value(tup_v, j),
                                               context.make_tuple(builder, types.UniTuple(i, 2),
                                                                  [i_v, context.get_constant(i, k)])))
                             for k in (0, 1)]
                    return context.make_tuple(builder, span, parts), span
                getitem = context.get_function(operator.getitem, signature(arr.dtype, arr, i))
                return getitem(builder, (builder.extract_value(tup_v, j), i_v)), arr.dtype

            def load(src):
                kind = src[0]
                if kind == "col":
                    return element(cols, cols_v, src[1])
                if kind == "opt":
                    v, t = element(cols, cols_v, src[1])
                    ok, _ = element(valids, valids_v, src[2])
                    some = context.make_optional_value(builder, t, v)
                    return builder.select(ok, some, context.make_optional_none(builder, t)), types.Optional(t)
                if kind == "res":
                    return results[src[1]][src[2]]
                if kind == "var":
                    return builder.load(slots[src[1]]), vtypes[src[1]]
                if kind == "par":
                    return builder.extract_value(params_v, src[1]), params.types[src[1]]
                loaded = [load(s) for s in src[1]]
                ty = types.Tuple(tuple(t for _, t in loaded))
                return context.make_tuple(builder, ty, [v for v, _ in loaded]), ty

            def assign(pairs):
                # Every source is loaded before any variable changes, so no update sees another's new value.
                loaded = [(var, load(src)) for var, src in pairs]
                for var, (v, t) in loaded:
                    builder.store(context.cast(builder, v, t, vtypes[var]), slots[var])

            def call(spec):
                loaded = [load(s) for s in spec.args]
                fnty = typeof(spec.fn)
                call_sig = fnty.get_call_type(context.typing_context, tuple(t for _, t in loaded), {})
                impl = context.get_function(fnty, call_sig)
                res = impl(builder, [context.cast(builder, v, t, want) for (v, t), want in zip(loaded, call_sig.args)])
                rt = call_sig.return_type
                parts = ([(builder.extract_value(res, k), rt.types[k]) for k in range(len(spec.dtypes))]
                         if spec.unpack else [(res, rt)])
                nullable = spec.nullable or (False,) * len(spec.dtypes)
                wants = [types.Optional(from_dtype(d)) if o else from_dtype(d) for d, o in zip(spec.dtypes, nullable)]
                results.append([(_cast(context, builder, v, t, w), w) for (v, t), w in zip(parts, wants)])

            def fork(x):
                v, t = load(x.test)
                if t == types.boolean:
                    with builder.if_else(v) as blocks:
                        for k, block in enumerate(blocks):
                            with block:
                                run(x.arms[k])
                                assign([(var, sources[k]) for var, sources in x.merges])
                    return
                end = builder.append_basic_block("fork.end")
                bad = builder.append_basic_block("fork.bad")
                switch = builder.switch(context.cast(builder, v, t, types.int64), bad)
                for k, arm in enumerate(x.arms):
                    block = builder.append_basic_block(f"fork.arm{k}")
                    switch.add_case(context.get_constant(types.int64, k), block)
                    builder.position_at_end(block)
                    run(arm)
                    assign([(var, sources[k]) for var, sources in x.merges])
                    builder.branch(end)
                builder.position_at_end(bad)
                context.call_conv.return_user_exc(builder, ArmOutOfRange, ("a branch condition picked a missing arm",))
                builder.position_at_end(end)

            def repeat(x):
                assign(x.carries)
                count = cgutils.alloca_once(builder, context.get_value_type(types.intp))
                builder.store(context.get_constant(types.intp, 0), count)
                head = builder.append_basic_block("loop.head")
                check = builder.append_basic_block("loop.check")
                work = builder.append_basic_block("loop.body")
                end = builder.append_basic_block("loop.end")
                builder.branch(head)
                builder.position_at_end(head)
                more = builder.icmp_signed("<", builder.load(count), context.get_constant(types.intp, x.max_iterations))
                builder.cbranch(more, check, end)
                builder.position_at_end(check)
                run(x.condition)
                v, t = load(x.test)
                builder.cbranch(context.cast(builder, v, t, types.boolean), work, end)
                builder.position_at_end(work)
                run(x.body)
                assign(x.updates)
                builder.store(builder.add(builder.load(count), context.get_constant(types.intp, 1)), count)
                builder.branch(head)
                builder.position_at_end(end)

            def run(block):
                for x in block:
                    if isinstance(x, Spec):
                        call(x)
                    elif isinstance(x, Fork):
                        fork(x)
                    else:
                        repeat(x)

            def store(k, value, ty):
                arr = outs.types[k]
                setitem = context.get_function(operator.setitem, signature(types.none, arr, i, ty))
                setitem(builder, (builder.extract_value(outs_v, k), i_v, value))

            run(program)
            masks = len(outputs)
            for k, src in enumerate(outputs):
                v, t = load(src)
                if isinstance(t, types.Optional):
                    opt = context.make_helper(builder, t, value=v)
                    zero = context.get_constant_null(t.type)
                    store(k, builder.select(opt.valid, opt.data, zero), t.type)
                    store(masks, opt.valid, types.boolean)
                    masks += 1
                else:
                    store(k, v, t)
            return context.get_dummy_value()

        return types.none(cols, valids, params, outs, i), codegen

    return body


def _cast(context, builder, value, ty, want):
    # A step that only ever raises returns `none`; its value is never stored.
    if ty == types.none and not isinstance(want, types.Optional):
        return context.get_constant_null(want)
    return context.cast(builder, value, ty, want)
