"""The indirect call: `int32 sm_match(void*, const uint8_t*, int64)` through
a raw address held in DATA (a `uint64` array argument), not a Python-level
global numba could specialise on.

Lifted from ../cfunc-pointer-interpreter/call_ptr.py (same `inttoptr` +
`call` codegen); only the signature differs. This is the whole reason the
kernel disk-caches: LLVM sees an integer argument being cast to a function
pointer, so nothing process-specific is baked into the compiled object.
"""
from __future__ import annotations

import llvmlite.ir as ll
from numba import types
from numba.extending import intrinsic

__all__ = ["call_match", "call_i32_i32", "load_i64"]


@intrinsic
def call_match(typingctx, addr_t, handle_t, ptr_t, len_t):
    """`int32 (*)(void*, const uint8_t*, int64)` -- `sm_match`'s shape.
    `addr` and `handle` are uint64 (the tables' dtype); `ptr` is a uint64
    byte address (`values.ctypes.data + offset`); `len` is int64."""
    ints = (types.uint64, types.int64, types.intp, types.uintp)
    if addr_t not in ints or handle_t not in ints or ptr_t not in ints or len_t not in ints:
        return None
    sig = types.int32(addr_t, handle_t, ptr_t, len_t)

    def codegen(context, builder, signature, args):
        addr, handle, ptr, length = args
        i8p = ll.IntType(8).as_pointer()
        fnty = ll.FunctionType(ll.IntType(32), [i8p, i8p, ll.IntType(64)])
        fnptr = builder.inttoptr(addr, fnty.as_pointer())
        h = builder.inttoptr(handle, i8p)
        p = builder.inttoptr(ptr, i8p)
        return builder.call(fnptr, [h, p, length])

    return sig, codegen


@intrinsic
def call_i32_i32(typingctx, addr_t, a_t, b_t):
    """`int32(int32,int32)` through a raw address -- the §V call-overhead probe shape."""
    if addr_t not in (types.uint64, types.int64) or a_t != types.int32 or b_t != types.int32:
        return None
    sig = types.int32(addr_t, types.int32, types.int32)

    def codegen(context, builder, signature, args):
        addr, a, b = args
        fnty = ll.FunctionType(ll.IntType(32), [ll.IntType(32), ll.IntType(32)])
        fnptr = builder.inttoptr(addr, fnty.as_pointer())
        return builder.call(fnptr, [a, b])

    return sig, codegen


@intrinsic
def load_i64(typingctx, addr_t):
    """`*(int64*)addr` -- a raw load through an integer address. This is how
    the walker reads a string column's Arrow offsets without holding an
    array object per column (see kernels.py's SHAPE NOTE)."""
    if addr_t not in (types.uint64, types.int64, types.intp, types.uintp):
        return None

    def codegen(context, builder, signature, args):
        p = builder.inttoptr(args[0], ll.IntType(64).as_pointer())
        return builder.load(p)

    return types.int64(addr_t), codegen
