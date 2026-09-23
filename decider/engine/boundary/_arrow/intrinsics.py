"""numba intrinsics that load from, and call through, raw addresses."""
from __future__ import annotations

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

_INT_TYPES = (types.uint64, types.int64, types.intp, types.uintp)
_I8P = ir.IntType(8).as_pointer()
_I64 = ir.IntType(64)


def _load_intrinsic(llvm_type, numba_type):
    @intrinsic
    def load(typingctx, addr_t):
        if not isinstance(addr_t, types.Integer):
            return None

        def codegen(context, builder, sig, args):
            return builder.load(builder.inttoptr(args[0], llvm_type.as_pointer()))

        return numba_type(addr_t), codegen

    return load


load_u8 = _load_intrinsic(ir.IntType(8), types.uint8)
load_i64 = _load_intrinsic(_I64, types.int64)
load_f64 = _load_intrinsic(ir.DoubleType(), types.float64)


def _as_i64(builder, value):
    return builder.sext(value, _I64) if value.type.width < 64 else value


@intrinsic
def call_gather(typingctx, fn_t, plan_t, i_t):
    """`sm_gather_row(plan, i)` through the function address `fn_t`."""
    if fn_t not in _INT_TYPES or plan_t not in _INT_TYPES or i_t not in _INT_TYPES:
        return None

    def codegen(context, builder, signature, args):
        fn, plan, i = args
        fnty = ir.FunctionType(ir.VoidType(), [_I8P, _I64])
        builder.call(builder.inttoptr(fn, fnty.as_pointer()),
                     [builder.inttoptr(plan, _I8P), _as_i64(builder, i)])
        return context.get_dummy_value()

    return types.none(fn_t, plan_t, i_t), codegen


@intrinsic
def call_get_string(typingctx, fn_t, view_t, i_t):
    """`sm_get_string(view, i)` through `fn_t`: `(data_addr, length)`, length -1 for a null."""
    if fn_t not in _INT_TYPES or view_t not in _INT_TYPES or i_t not in _INT_TYPES:
        return None
    ret = types.UniTuple(types.int64, 2)

    def codegen(context, builder, signature, args):
        fn, view, i = args
        fnty = ir.FunctionType(_I64, [_I8P, _I64, _I8P.as_pointer()])
        slot = cgutils.alloca_once(builder, _I8P)
        length = builder.call(builder.inttoptr(fn, fnty.as_pointer()),
                              [builder.inttoptr(view, _I8P), _as_i64(builder, i), slot])
        data = builder.ptrtoint(builder.load(slot), _I64)
        return context.make_tuple(builder, ret, [data, length])

    return ret(fn_t, view_t, i_t), codegen
