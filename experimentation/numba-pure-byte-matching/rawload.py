"""Raw loads through integer addresses -- the ONLY non-Python-level thing
this strand uses. No C, no ctypes call, no new dependency: an `@intrinsic`
that emits an LLVM `inttoptr` + `load`. `load_i64` is lifted verbatim from
../numba-c-string-matching/call_ptr.py; `load_u8` and `load_u64` are the
same three lines at other widths.

Why raw loads at all: the sibling strand's refcount ablation showed that a
per-row njit function taking ARRAYS emits NRT incref/decref per row
(~365 ns/row). Reading the Arrow buffers through plain integers keeps every
per-row input a scalar, which is also the shape decider2's `walk_tree`
already has (a tuple of scalars).
"""
from __future__ import annotations

import llvmlite.ir as ll
from numba import types
from numba.extending import intrinsic

_INTS = (types.uint64, types.int64, types.intp, types.uintp)


@intrinsic
def load_u8(typingctx, addr_t):
    """`*(uint8*)addr`."""
    if addr_t not in _INTS:
        return None

    def codegen(context, builder, signature, args):
        p = builder.inttoptr(args[0], ll.IntType(8).as_pointer())
        return builder.load(p)

    return types.uint8(addr_t), codegen


@intrinsic
def load_i64(typingctx, addr_t):
    """`*(int64*)addr` -- Arrow offsets."""
    if addr_t not in _INTS:
        return None

    def codegen(context, builder, signature, args):
        p = builder.inttoptr(args[0], ll.IntType(64).as_pointer())
        return builder.load(p)

    return types.int64(addr_t), codegen


@intrinsic
def load_u64(typingctx, addr_t):
    """`*(uint64*)addr`, unaligned -- eight bytes at once, for a word-wise
    memcmp. Marked align 1 so LLVM never assumes alignment the Arrow
    buffer does not promise."""
    if addr_t not in _INTS:
        return None

    def codegen(context, builder, signature, args):
        p = builder.inttoptr(args[0], ll.IntType(64).as_pointer())
        return builder.load(p, align=1)

    return types.uint64(addr_t), codegen
