from __future__ import annotations

import operator
from typing import Any, Callable, NamedTuple

import numpy as np
from numba import from_dtype, njit, typeof
from numba.core import types
from numba.core.typing import signature
from numba.extending import intrinsic


class Spec(NamedTuple):
    """One call inside a fused kernel.

    `args` holds one source per positional argument of `fn`:
    `("col", j)` is element i of input column j; `("opt", j, m)` is the same
    element, or `None` where validity mask m is false; `("res", s, k)` is
    output k of call s of this kernel; `("par", p)` is `params[p]`;
    `("row", (source, ...))` is a tuple of sources. `dtypes` are the declared
    dtypes of the outputs; `unpack` means `fn` returns a tuple of them.
    """

    key: str
    fn: Callable
    args: tuple
    dtypes: tuple[np.dtype, ...]
    unpack: bool


# ponytail: unbounded like the step dispatchers; evict with them if sessions ever churn through edits.
_KERNELS: dict[tuple, Callable] = {}


def fused_kernel(specs: tuple[Spec, ...], outputs: tuple[tuple[int, int], ...]) -> Callable:
    """One njit kernel `kernel(n, cols, valids, params, outs)` running `specs` in order for each of n rows.

    `outputs[k] = (s, j)` writes output j of call s into `outs[k]`. Every
    value is cast to its declared dtype when produced, so a fused value
    matches the one a single-call kernel stores. Kernels are shared by
    content: the same specs give the same kernel, whatever the nodes are called.

    Example::

        kernel = fused_kernel((Spec(key, fn, (("col", 0),), (np.dtype("f8"),), False),), ((0, 0),))
        kernel(n, (x,), (), (), (out,))
    """
    key = (tuple((s.key, s.args, s.dtypes, s.unpack) for s in specs), outputs)
    kernel = _KERNELS.get(key)
    if kernel is None:
        body = _row_body(specs, outputs)

        def run(n, cols, valids, params, outs):
            for i in range(n):
                body(cols, valids, params, outs, i)

        kernel = _KERNELS[key] = njit(nogil=True)(run)
    return kernel


def _row_body(specs: tuple[Spec, ...], outputs: tuple[tuple[int, int], ...]) -> Any:
    # Lowered straight to IR: each call goes through the same
    # get_call_type/get_function/cast sequence numba uses for a direct call in
    # jitted source, and results stay SSA values for the calls after it.
    @intrinsic
    def body(typingctx, cols, valids, params, outs, i):
        def codegen(context, builder, sig, args):
            cols_v, valids_v, params_v, outs_v, i_v = args
            results: list[list[tuple[Any, Any]]] = []

            def element(tup, tup_v, j):
                arr = tup.types[j]
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
                if kind == "par":
                    return builder.extract_value(params_v, src[1]), params.types[src[1]]
                loaded = [load(s) for s in src[1]]
                ty = types.Tuple(tuple(t for _, t in loaded))
                return context.make_tuple(builder, ty, [v for v, _ in loaded]), ty

            for spec in specs:
                loaded = [load(s) for s in spec.args]
                fnty = typeof(spec.fn)
                call = fnty.get_call_type(context.typing_context, tuple(t for _, t in loaded), {})
                impl = context.get_function(fnty, call)
                res = impl(builder, [context.cast(builder, v, t, want) for (v, t), want in zip(loaded, call.args)])
                rt = call.return_type
                parts = ([(builder.extract_value(res, k), rt.types[k]) for k in range(len(spec.dtypes))]
                         if spec.unpack else [(res, rt)])
                wants = [from_dtype(d) for d in spec.dtypes]
                results.append([(context.cast(builder, v, t, w), w) for (v, t), w in zip(parts, wants)])

            for k, (s, j) in enumerate(outputs):
                arr = outs.types[k]
                v, t = results[s][j]
                setitem = context.get_function(operator.setitem, signature(types.none, arr, i, t))
                setitem(builder, (builder.extract_value(outs_v, k), i_v, v))
            return context.get_dummy_value()

        return types.none(cols, valids, params, outs, i), codegen

    return body
