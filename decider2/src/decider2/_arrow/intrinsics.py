"""Raw-address loads for numba kernels — pure LLVM (`inttoptr` + `load`),
no dynamic global, no extension: importing this module never loads
`decider2._arrow._nashim`.

Split out of `_shim.py` so `decider2.trees.interpreter` can read a byte at
an address (a string feature's span, a pattern's bytes) without making
`import decider2.trees` depend on the compiled shim. `_shim` re-exports
these names, so `decider2._arrow.load_u8` is the same object either way.
Each takes the address as an integer VALUE, so the compiled code contains
no process-specific constant and disk-caches (EXPERIMENTS.md §W).
"""
from __future__ import annotations

from llvmlite import ir
from numba import types
from numba.extending import intrinsic

__all__ = ["load_u8", "load_i64", "load_f64"]


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
load_i64 = _load_intrinsic(ir.IntType(64), types.int64)
load_f64 = _load_intrinsic(ir.DoubleType(), types.float64)
