"""The raw mechanism this whole experiment is about: calling a
`@numba.cfunc`'s `.address` from inside another `@njit` function through an
indirect call, where the address is DATA -- read from a numpy array at a
runtime index -- not a Python-level global numba can specialise on at
compile time.

This is a fourth, distinct calling mechanism from the three EXPERIMENTS.md
already measured:

  - njit -> njit direct call            -- fully inlinable, §V's 0.00 ns
  - njit -> ctypes-global extern "C"    -- §V/§T's mechanism: the ctypes
    function OBJECT is a Python-level global, numba specialises the call
    site on it at compile time, the address is baked in as an LLVM
    constant. §V measured this at 2.4-3.84 ns/call.
  - numba.typed.List[FunctionType]      -- §S's mechanism: dispatch through
    numba Dispatcher VALUES stored in a typed container. 28-41x slower than
    codegen, 14-20x slower than §Q's array interpreter. LOSES badly.
  - raw address, runtime-indexed table  -- THIS mechanism. `@cfunc` gives a
    real C function pointer (an integer, `.address`); we store N of them in
    a `np.uint64` array and, at each call site, read `ptr_table[k]` for a
    runtime `k` and perform `inttoptr` + `call` via an `@intrinsic`. Nothing
    about the callee is known to LLVM until runtime. This is what a generic
    "node says: call step k" interpreter actually needs -- a real pointer
    table, not a statically-resolvable global.

Built once, reused by every other script in this experiment (the interpreter
in `interpreter.py`, the call-overhead benchmark in `bench_call_overhead.py`).
"""
from __future__ import annotations

from numba import types
from numba.extending import intrinsic
import llvmlite.ir as ll

__all__ = [
    "call_f64_f64", "call_f64_f64_f64", "call_f64_3args", "call_bool_f64f64f64f64",
    "call_i32_i32", "call_bool_1arg", "call_bool_2args", "call_bool_3args",
]


@intrinsic
def call_f64_f64(typingctx, addr_t, x_t):
    """Indirect call through a raw address to a `float64(float64)` cfunc.
    `addr_t` must be an integer type (the pointer table's dtype); `x_t` must
    be float64. Used for a plain one-input step."""
    if addr_t not in (types.uint64, types.int64) or x_t != types.float64:
        return None
    sig = types.float64(addr_t, types.float64)

    def codegen(context, builder, signature, args):
        addr_val, x_val = args
        fnty = ll.FunctionType(ll.DoubleType(), [ll.DoubleType()])
        fnptr = builder.inttoptr(addr_val, fnty.as_pointer())
        return builder.call(fnptr, [x_val])

    return sig, codegen


@intrinsic
def call_f64_f64_f64(typingctx, addr_t, x_t, y_t):
    """Indirect call through a raw address to a `float64(float64, float64)`
    cfunc -- a two-input step (e.g. a branch condition reading two carried
    values, or a step combining a row value with a carry slot)."""
    if addr_t not in (types.uint64, types.int64) or x_t != types.float64 or y_t != types.float64:
        return None
    sig = types.float64(addr_t, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        addr_val, x_val, y_val = args
        fnty = ll.FunctionType(ll.DoubleType(), [ll.DoubleType(), ll.DoubleType()])
        fnptr = builder.inttoptr(addr_val, fnty.as_pointer())
        return builder.call(fnptr, [x_val, y_val])

    return sig, codegen


@intrinsic
def call_f64_3args(typingctx, addr_t, a_t, b_t, c_t):
    """Indirect call through a raw address to a `float64(float64, float64,
    float64)` cfunc -- a body step reading a carry plus two other values
    (e.g. `fast_bump(term_cap, loop_idx, jump)` from decider2's own
    `playground_control_flow.py`, reused verbatim for item 3's identical-
    work comparison)."""
    for t in (a_t, b_t, c_t):
        if t != types.float64:
            return None
    if addr_t not in (types.uint64, types.int64):
        return None
    sig = types.float64(addr_t, types.float64, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        addr_val, a, b, c = args
        fnty = ll.FunctionType(ll.DoubleType(), [ll.DoubleType()] * 3)
        fnptr = builder.inttoptr(addr_val, fnty.as_pointer())
        return builder.call(fnptr, [a, b, c])

    return sig, codegen


@intrinsic
def call_bool_1arg(typingctx, addr_t, a_t):
    """Indirect call through a raw address to a `bool(float64)` cfunc --
    e.g. `is_high_income(high_income)`."""
    if addr_t not in (types.uint64, types.int64) or a_t != types.float64:
        return None
    sig = types.boolean(addr_t, types.float64)

    def codegen(context, builder, signature, args):
        addr_val, a = args
        fnty = ll.FunctionType(ll.IntType(1), [ll.DoubleType()])
        fnptr = builder.inttoptr(addr_val, fnty.as_pointer())
        return builder.call(fnptr, [a])

    return sig, codegen


@intrinsic
def call_bool_2args(typingctx, addr_t, a_t, b_t):
    """`bool(float64, float64)` cfunc -- e.g.
    `fast_track_should_continue(loop_idx, micro_steps)`."""
    if addr_t not in (types.uint64, types.int64) or a_t != types.float64 or b_t != types.float64:
        return None
    sig = types.boolean(addr_t, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        addr_val, a, b = args
        fnty = ll.FunctionType(ll.IntType(1), [ll.DoubleType()] * 2)
        fnptr = builder.inttoptr(addr_val, fnty.as_pointer())
        return builder.call(fnptr, [a, b])

    return sig, codegen


@intrinsic
def call_bool_3args(typingctx, addr_t, a_t, b_t, c_t):
    """`bool(float64, float64, float64)` cfunc -- e.g.
    `term_still_short(term_cap, loop_idx, floor)` /
    `instalment_fits(term_cap, loop_idx, ceiling-adjusted)`."""
    for t in (a_t, b_t, c_t):
        if t != types.float64:
            return None
    if addr_t not in (types.uint64, types.int64):
        return None
    sig = types.boolean(addr_t, types.float64, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        addr_val, a, b, c = args
        fnty = ll.FunctionType(ll.IntType(1), [ll.DoubleType()] * 3)
        fnptr = builder.inttoptr(addr_val, fnty.as_pointer())
        return builder.call(fnptr, [a, b, c])

    return sig, codegen


@intrinsic
def call_i32_i32(typingctx, addr_t, a_t, b_t):
    """Indirect call through a raw address to an `int32(int32, int32)`
    cfunc. Exists only for the call-overhead signature-sensitivity probe in
    `bench_call_overhead.py` -- decider2 steps do not use this shape (money
    is scaled int64, doc 00 §2; everything else is float64/bool at the
    kernel boundary, doc 05 §1.5), but §V's 2.4-3.84 ns figure WAS measured
    on exactly this shape (Rust's `trivial(i32,i32)->i32`), so reproducing
    it is what makes the comparison honest."""
    if addr_t not in (types.uint64, types.int64) or a_t != types.int32 or b_t != types.int32:
        return None
    sig = types.int32(addr_t, types.int32, types.int32)

    def codegen(context, builder, signature, args):
        addr_val, a_val, b_val = args
        fnty = ll.FunctionType(ll.IntType(32), [ll.IntType(32), ll.IntType(32)])
        fnptr = builder.inttoptr(addr_val, fnty.as_pointer())
        return builder.call(fnptr, [a_val, b_val])

    return sig, codegen


@intrinsic
def call_bool_f64f64f64f64(typingctx, addr_t, a_t, b_t, c_t, d_t):
    """Indirect call through a raw address to a `bool(float64, float64,
    float64, float64)` cfunc -- a branch/leaf CONDITION step reading up to 4
    row features, matching the tree interpreter's feature arity in
    `tree_shapes.py` (reused from `tree-codegen-vs-interpreted`)."""
    for t in (a_t, b_t, c_t, d_t):
        if t != types.float64:
            return None
    if addr_t not in (types.uint64, types.int64):
        return None
    sig = types.boolean(addr_t, types.float64, types.float64, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        addr_val, a, b, c, d = args
        fnty = ll.FunctionType(ll.IntType(1), [ll.DoubleType()] * 4)
        fnptr = builder.inttoptr(addr_val, fnty.as_pointer())
        return builder.call(fnptr, [a, b, c, d])

    return sig, codegen
