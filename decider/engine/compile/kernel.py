from __future__ import annotations

import operator
from enum import Enum
from typing import Any, Callable, NamedTuple

import numpy as np
from numba import from_dtype, njit, typeof
from numba.core import cgutils, types
from numba.core.typing import signature
from numba.extending import intrinsic

from decider.engine.compile.rows import rows_class
from decider.engine.compile.span import SPAN
from decider.engine.trace.envelope import ITER_SHIFT, Kind, pack


class SrcKind(str, Enum):
    """Tag for the first element of a source tuple passed to the kernel IR builder."""
    COL = "col"    # element at column j
    OPT = "opt"    # nullable element: col[i] or None where mask is false
    RES = "res"    # output k of call s earlier in this kernel
    VAR = "var"    # variable set by a Fork or Repeat
    PAR = "par"    # element of the flattened params array
    RAG = "rag"    # ragged column: row i's items sliced from flat arrays
    SINK = "sink"  # a growable output buffer, shared by every row (not indexed by i)
    ROW = "row"    # tuple of sub-sources packed into a single argument


def _slice(arr, lo, hi):
    return arr[lo:hi]


class Spec(NamedTuple):
    """One call inside a fused kernel.

    `args` holds one source per positional argument of `fn`:
    `("col", j)` is element i of input column j; `("opt", j, m)` is the same
    element, or `None` where validity mask m is false; `("res", s, k)` is
    output k of call s of this kernel; `("var", v)` is variable v (set by a
    `Fork` or `Repeat`); `("par", p)` is `params[p]`;
    `("row", (source, ...))` is a tuple of sources; `("rag", schema, b)` is
    row i's items, sliced out of `flats[b:]` (`lo`, `hi`, then one flat array
    per field of `schema`); `("sink", b)` is `sinks[b]`, a growable output
    buffer shared by every row of this call. `dtypes` are the declared dtypes of the outputs;
    `unpack` means `fn` returns a tuple of them. `nullable[k]` marks an output
    declared `T | None`: it may be `None`. `structs[k]` marks a `Struct[Item]`
    output, whose `fn` returns a tuple of its fields.
    """

    key: str
    fn: Callable
    args: tuple
    dtypes: tuple[np.dtype, ...]
    unpack: bool
    nullable: tuple[bool, ...] = ()
    structs: tuple[bool, ...] = ()


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


def fused_kernel(program: tuple, outputs: tuple[tuple, ...], variables: tuple[np.dtype, ...] = (),
                 trace_refs: tuple[int, ...] | None = None) -> Callable:
    """One njit kernel `kernel(n, cols, valids, params, flats, sinks, outs)` running `program` for each of n rows.

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

    With `trace_refs`, the kernel also captures a decision trace: it takes two
    extra int64 arrays (`header`, `cursor`) and an offsets array, and writes
    one `Kind.STEP` event per call, in call order, carrying the call's ref
    (from `trace_refs`, in the same depth-first order) and its branch-arm /
    loop-iteration context. The signature then is
    `kernel(n, cols, valids, params, flats, sinks, outs, header, cursor, offsets)`.

    Example::

        kernel = fused_kernel((Spec(key, fn, (("col", 0),), (np.dtype("f8"),), False),), (("res", 0, 0),))
        kernel(n, (x,), (), (), (), (), (out,))
    """
    trace = trace_refs is not None
    key = (_key(program), outputs, variables, trace, trace_refs)
    kernel = _KERNELS.get(key)
    if kernel is None:
        body = _row_body(program, outputs, variables, trace, trace_refs)

        if trace:

            def run(n, cols, valids, params, flats, sinks, outs, header, cursor, offsets):
                for i in range(n):
                    body(cols, valids, params, flats, sinks, outs, i, header, cursor)
                    offsets[i] = cursor[0]
        else:

            def run(n, cols, valids, params, flats, sinks, outs):
                for i in range(n):
                    body(cols, valids, params, flats, sinks, outs, i)

        kernel = _KERNELS[key] = njit(nogil=True)(run)
    return kernel


def _key(program: tuple) -> tuple:
    out = []
    for x in program:
        if isinstance(x, Spec):
            out.append((x.key, x.args, x.dtypes, x.unpack, x.nullable, x.structs))
        elif isinstance(x, Fork):
            out.append(("fork", x.test, tuple(_key(a) for a in x.arms), x.merges))
        else:
            out.append(("repeat", x.carries, _key(x.condition), x.test, _key(x.body), x.updates, x.max_iterations))
    return tuple(out)


def _row_body(program: tuple, outputs: tuple[tuple, ...], variables: tuple[np.dtype, ...],
              trace: bool = False, trace_refs: tuple[int, ...] = ()) -> Any:
    # Lowered straight to IR: each call goes through the same
    # get_call_type/get_function/cast sequence numba uses for a direct call in
    # jitted source, and results stay SSA values for the calls after it.
    # Forks and loops are plain LLVM blocks; values crossing them live in
    # stack slots, which LLVM promotes back to registers. With `trace`, each
    # call also writes one `Kind.STEP` event into `header` (its ref in
    # `trace_refs`, its arm/iteration context) and advances `cursor`.
    i64 = types.int64
    ip = types.intp

    def codegen(context, builder, sig, args):
        # The argument types: codegen is shared by the traced and untraced
        # intrinsics, so it reads the types from the signature, not from a
        # typing function's parameters.
        cols, valids, params, flats, sinks, outs, i = sig.args[:7]
        header = sig.args[7] if trace else None
        cursor = sig.args[8] if trace else None
        cols_v, valids_v, params_v, flats_v, sinks_v, outs_v, i_v = args[:7]
        header_v = args[7] if trace else None
        cursor_v = args[8] if trace else None
        results: list[list[tuple[Any, Any]]] = []
        vtypes = [from_dtype(d) for d in variables]
        # A record scalar is a pointer to its bytes; a carried record's slot holds the bytes so
        # the pointer stays valid past the iteration that produced it.
        slots = [cgutils.alloca_once(builder, context.get_data_type(t) if isinstance(t, types.Record)
                                     else context.get_value_type(t)) for t in vtypes]

        def element(tup, tup_v, j):
            arr = tup.types[j]
            if arr.ndim == 2:
                getitem = context.get_function(operator.getitem, signature(arr.dtype, arr, types.UniTuple(i, 2)))
                parts = [getitem(builder, (builder.extract_value(tup_v, j),
                                           context.make_tuple(builder, types.UniTuple(i, 2),
                                                              [i_v, context.get_constant(i, k)])))
                         for k in (0, 1)]
                # A `bytes` column: row i is its `(address, byte length)` span.
                return context.make_tuple(builder, SPAN, parts), SPAN
            getitem = context.get_function(operator.getitem, signature(arr.dtype, arr, i))
            return getitem(builder, (builder.extract_value(tup_v, j), i_v)), arr.dtype

        def rag(schema, base):
            # Row i's items: each flat field array sliced `lo[i]:hi[i]`. The field types come
            # from the arrays themselves, never from a probe: read straight out of Arrow they
            # are `readonly array(...)`, which is a type of its own.
            lo, index = element(flats, flats_v, base)
            hi, _ = element(flats, flats_v, base + 1)
            fields = [flats.types[base + 2 + k] for k in range(len(schema))]
            ty = types.NamedTuple(fields, rows_class(schema))
            views = [context.compile_internal(builder, _slice, signature(at, at, index, index),
                                              [builder.extract_value(flats_v, base + 2 + k), lo, hi])
                     for k, at in enumerate(fields)]
            return context.make_tuple(builder, ty, views), ty

        def load(src):
            kind = src[0]
            if kind == SrcKind.COL:
                # Plain column: cols[j][i]
                return element(cols, cols_v, src[1])
            if kind == SrcKind.OPT:
                # Nullable column: cols[j][i] wrapped as Optional, or None where mask[m][i] is false
                v, t = element(cols, cols_v, src[1])
                ok, _ = element(valids, valids_v, src[2])
                some = context.make_optional_value(builder, t, v)
                return builder.select(ok, some, context.make_optional_none(builder, t)), types.Optional(t)
            if kind == SrcKind.RES:
                # Earlier call result: results[call_index][output_index]
                return results[src[1]][src[2]]
            if kind == SrcKind.VAR:
                # Fork/Repeat variable: load from its stack slot (records need a pointer cast)
                t = vtypes[src[1]]
                if isinstance(t, types.Record):
                    return builder.bitcast(slots[src[1]], context.get_value_type(t)), t
                return builder.load(slots[src[1]]), t
            if kind == SrcKind.PAR:
                # Flattened params array element
                return builder.extract_value(params_v, src[1]), params.types[src[1]]
            if kind == SrcKind.RAG:
                # Ragged columnar input: slice row i's items out of flat arrays
                return rag(src[1], src[2])
            if kind == SrcKind.SINK:
                # Growable output buffer: one value shared by every row, never indexed by i
                return builder.extract_value(sinks_v, src[1]), sinks.types[src[1]]
            # SrcKind.ROW: pack multiple sub-sources into a single tuple argument
            loaded = [load(s) for s in src[1]]
            ty = types.Tuple(tuple(t for _, t in loaded))
            return context.make_tuple(builder, ty, [v for v, _ in loaded]), ty

        def assign(pairs):
            # Every source is loaded before any variable changes, so no update sees another's new value.
            loaded = [(var, load(src)) for var, src in pairs]
            for var, (v, t) in loaded:
                vt = vtypes[var]
                if isinstance(vt, types.Record):
                    # Copy the record's bytes, not its pointer: the source's buffer may be
                    # an alloca in the block that produced it.
                    src = context.cast(builder, v, t, vt)
                    builder.store(builder.load(src), slots[var])
                else:
                    builder.store(context.cast(builder, v, t, vt), slots[var])

        def build_record(fields, rec_ty):
            # A `Struct[Item]` step returns a tuple of its fields; pack them into a record
            # scalar (a pointer to a byte buffer the record type names), which flows through
            # carries and reads like any record.
            data_ty = context.get_data_type(rec_ty)
            ptr = cgutils.alloca_once(builder, data_ty)
            for j, (name, ftype) in enumerate(rec_ty.members):
                field = context.cast(builder, fields[j][0], fields[j][1], ftype)
                dest = cgutils.get_record_member(builder, ptr, rec_ty.offset(name),
                                                 context.get_data_type(ftype))
                context.pack_value(builder, ftype, field, dest)
            return builder.bitcast(ptr, context.get_value_type(rec_ty)), rec_ty

        if trace:

            def emit(value):
                # header[cursor] = value; cursor += 1, with no Python allocation on the hot path.
                zero = context.get_constant(ip, 0)
                load_c = context.get_function(operator.getitem, signature(i64, cursor, ip))
                idx = load_c(builder, (cursor_v, zero))
                set_h = context.get_function(operator.setitem, signature(types.none, header, ip, i64))
                set_h(builder, (header_v, idx, value))
                set_c = context.get_function(operator.setitem, signature(types.none, cursor, ip, i64))
                set_c(builder, (cursor_v, zero, builder.add(idx, context.get_constant(i64, 1))))

        def call(spec, arm, iteration):
            loaded = [load(s) for s in spec.args]
            fnty = typeof(spec.fn)
            call_sig = fnty.get_call_type(context.typing_context, tuple(t for _, t in loaded), {})
            impl = context.get_function(fnty, call_sig)
            res = impl(builder, [context.cast(builder, v, t, want) for (v, t), want in zip(loaded, call_sig.args)])
            rt = call_sig.return_type
            parts = ([(builder.extract_value(res, k), rt.types[k]) for k in range(len(spec.dtypes))]
                     if spec.unpack else [(res, rt)])
            nullable = spec.nullable or (False,) * len(spec.dtypes)
            structs = spec.structs or (False,) * len(spec.dtypes)
            wants = [types.Optional(from_dtype(d)) if o else from_dtype(d) for d, o in zip(spec.dtypes, nullable)]
            # A struct output's step returns a tuple of its fields (pack them into a record), or an
            # already-record value (copy it); a non-tuple, non-record return (a dict) cannot be
            # stored, so the kernel falls back.
            def struct_value(v, t, w):
                if isinstance(t, types.BaseTuple):
                    return build_record([(builder.extract_value(v, j), t.types[j])
                                         for j in range(len(t.types))], w)
                return v, w

            results.append([struct_value(v, t, w) if f else (_cast(context, builder, v, t, w), w)
                            for (v, t), w, f in zip(parts, wants, structs)])
            if trace:
                # The STEP event: this call's ref, its arm and iteration context.
                base = context.get_constant(i64, pack(Kind.STEP, trace_refs[len(results) - 1], arm, 0))
                if iteration is None:
                    emit(base)
                else:
                    emit(builder.or_(base, builder.shl(iteration, context.get_constant(i64, ITER_SHIFT))))

        def fork(x, iteration):
            v, t = load(x.test)
            if t == types.boolean:
                with builder.if_else(v) as blocks:
                    for k, block in enumerate(blocks):
                        with block:
                            run(x.arms[k], arm=k + 1, iteration=iteration)
                            assign([(var, sources[k]) for var, sources in x.merges])
                return
            end = builder.append_basic_block("fork.end")
            bad = builder.append_basic_block("fork.bad")
            switch = builder.switch(context.cast(builder, v, t, types.int64), bad)
            for k, arm in enumerate(x.arms):
                block = builder.append_basic_block(f"fork.arm{k}")
                switch.add_case(context.get_constant(types.int64, k), block)
                builder.position_at_end(block)
                run(arm, arm=k + 1, iteration=iteration)
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
            iteration = builder.add(builder.load(count), context.get_constant(ip, 1))
            run(x.condition, arm=0, iteration=iteration if trace else None)
            v, t = load(x.test)
            builder.cbranch(context.cast(builder, v, t, types.boolean), work, end)
            builder.position_at_end(work)
            run(x.body, arm=0, iteration=iteration if trace else None)
            assign(x.updates)
            builder.store(builder.add(builder.load(count), context.get_constant(types.intp, 1)), count)
            builder.branch(head)
            builder.position_at_end(end)

        def run(block, arm=0, iteration=None):
            for x in block:
                if isinstance(x, Spec):
                    call(x, arm, iteration)
                elif isinstance(x, Fork):
                    fork(x, iteration)
                else:
                    repeat(x)

        def store(k, value, ty):
            arr = outs.types[k]
            setitem = context.get_function(operator.setitem, signature(types.none, arr, i, ty))
            setitem(builder, (builder.extract_value(outs_v, k), i_v, value))

        def store_record(k, value, ty):
            # `value` is a record scalar; `outs[k]` is the record array it is stored into.
            arr = outs.types[k]
            setitem = context.get_function(operator.setitem, signature(types.none, arr, i, arr.dtype))
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
            elif isinstance(outs.types[k].dtype, types.Record):
                store_record(k, v, t)
            else:
                store(k, v, t)
        return context.get_dummy_value()

    if trace:

        @intrinsic
        def body(typingctx, cols, valids, params, flats, sinks, outs, i, header, cursor):
            return types.none(cols, valids, params, flats, sinks, outs, i, header, cursor), codegen

    else:

        @intrinsic
        def body(typingctx, cols, valids, params, flats, sinks, outs, i):
            return types.none(cols, valids, params, flats, sinks, outs, i), codegen

    return body


def _cast(context, builder, value, ty, want):
    # A step that only ever raises returns `none`; its value is never stored.
    if ty == types.none and not isinstance(want, types.Optional):
        return context.get_constant_null(want)
    return context.cast(builder, value, ty, want)
