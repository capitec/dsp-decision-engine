# Packed branches and loops: one kernel with LLVM control flow

`engine/compile/packed.py` and the `Fork`/`Repeat` items of
`engine/compile/kernel.py`. Choices fused mode, sessions and later tasks
build on.

**Decision:** In fused mode a branch or loop whose every call is a plain
scalar call (nested branches and loops included) compiles into one kernel.
Each row takes its own arm, or its own iterations, inside the kernel, and an
untaken arm costs nothing. The control flow is emitted as LLVM blocks by the
same intrinsic that fuses straight sequences: a `Fork` is an `if/else` (bool
condition) or a `switch` (int condition), a `Repeat` is a counted loop. Values
crossing a block (merged names, carried names) live in stack slots that LLVM
promotes back to registers; everything else stays an SSA value. No source is
generated.

**Why not decider2's approach:** decider2 walked an opcode program with a
float64 register array per row and called each step through an adapter
closure. That is data-driven but pays for dispatch and float round trips per
call. Emitting the blocks directly gives the straight-line kernel's code
inside each arm. Measured (`benchmarks/control_flow_vs_decider2.py`, fused,
CPython 3.14, dev box, 2026-09-23; outputs identical to decider2's):

| engine | branch, 1M rows | loop (<= 360 iterations), 200k rows |
|---|---|---|
| decider2 fused | 3.3-3.5M rows/s | 0.64M rows/s |
| decider fused (packed) | 41.6-44.3M rows/s | 6.33M rows/s |
| decider stepped (Python-driven) | 14.8M rows/s | 0.81M rows/s |

`score()` through that branch: fused p50 23 µs, stepped 66 µs.

**Packing rules** (anything else runs the Python-driven path of stepped
mode, with its per-call kernels, and gives the same answers):
- every call inside is `scalar` and numba compiles it alone; no row or
  frame nodes;
- no call inside has a nullable (`T | None`) output;
- every merged or carried name is `float`, `int` or `bool`, with one dtype
  across the arms or the carry's versions;
- nothing made inside but what the branch merges or the loop carries is a
  pipeline output (a value emitted by `name@path` from inside stays
  unpacked, since a packed kernel never stores it);
- with lazy params validation, no call except the branch's or loop's own
  condition has params (see `lazy-validation-fused.md`).

**At run time** the driver takes the unpacked path for one launch when:
- a value the kernel only copies (a merge's earlier value, a carry's
  initial one) has a null on a row it runs, since the packed kernel has no
  validity for it;
- or a REQUIRED input of a call that may not run has a null, since
  interpreted mode only fails on rows that reach that call.

An int condition out of range raises `ArmOutOfRange` in the kernel; the
driver re-runs the branch unpacked, which raises the usual error naming the
branch, the arm and the row count. A kernel numba can't build as a whole is
dropped and the branch or loop runs unpacked from then on.

**Sessions:** a packed branch or loop is one checkpoint pair with its own
path, registered like a fused kernel: `Paused.kernel` is the branch or loop
followed by every call inside, a breakpoint on any of them pauses before it,
and values made inside raise the stepped-mode `KeyError` (unless a launch
took the unpacked path and stored them). Every packable branch and loop gets
a kernel, nested ones too, because an outer one may take the unpacked path
at run time.
