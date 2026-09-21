"""Item 2 -- a generic control-flow interpreter over flat arrays, no
generated source, ever. One `@njit` kernel (`run_program`) walks a program
described entirely as DATA (parallel numpy arrays, one row per node) and
calls each step through `ptr_table[step_idx[pc]]` -- a `@numba.cfunc`
address selected at runtime, via the raw-pointer mechanism in `call_ptr.py`.

**Opcode set.** Small and closed -- everything Branch/Loop/a straight-line
sequence needs, nothing more:

  STEP1 / STEP3   call a float64(float64) / float64(float64,float64,float64)
                  cfunc, write the result into `regs[dest]`, fall through.
  COND1/2/3       call a bool(...) cfunc (1/2/3 float64 args), branch to
                  `next[pc]` if true, `alt[pc]` if false. This is BOTH
                  Branch's condition test and a Loop's `should_continue`
                  check -- a loop head is just a COND node whose false edge
                  points past the loop and whose true edge enters the body.
  SET_ZERO        `regs[dest] = 0.0` -- loop-counter init, run once on
                  entering a loop (never on the back-edge).
  INCR            `regs[dest] += 1.0` -- loop-counter increment, run once
                  per iteration on the back-edge (never on entry).
  JUMP            unconditional `pc = next[pc]` -- merges Branch arms back
                  into one control path, or closes a loop's back-edge.
  LEAF            `out[row] = regs[dest]`, stop this row.

**Registers, not a stack.** Each row gets a small fixed-size float64
`regs` array: carried values (a Loop's `carries`), loop counters
(`loop_idx`), row inputs (leaf columns) and PARAMS all live in the same
array, at fixed indices decided when the program is built. Params occupy
their own slots, written ONCE before the row loop (never touched inside
it) and copied fresh into every row's `regs` via `regs = param_template.
copy()` -- which is also this build's answer to item 6 (`params_check.py`):
retuning a threshold is `param_template[k] = new_value`, no compile event
of any kind, not even a new specialisation.

**No generated source at any point** -- `run_program` is ONE `@njit`
function, compiled once, reused for every program (every tree, every
Branch/Loop shape, any nesting depth) by swapping which arrays are passed
in. Building a NEW program is building new numpy arrays in Python, which is
`build_program()` below -- a data transformation, not a compile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from call_ptr import (
    call_bool_1arg, call_bool_2args, call_bool_3args,
    call_f64_f64, call_f64_3args,
)

# --- opcodes ---
STEP1 = 0
STEP3 = 1
COND1 = 2
COND2 = 3
COND3 = 4
SET_ZERO = 5
INCR = 6
JUMP = 7
LEAF = 8

NO_ARG = -1


@dataclass(frozen=True)
class Program:
    """One control-flow program, entirely as flat int32 arrays -- the
    "source text" a real decider2 Branch/Loop would instead emit as a
    `.py` file. `n_regs` sizes the per-row register scratch; `start_pc`
    is where every row begins."""

    op: np.ndarray
    step_idx: np.ndarray
    arg0: np.ndarray
    arg1: np.ndarray
    arg2: np.ndarray
    dest: np.ndarray
    next_: np.ndarray
    alt: np.ndarray
    n_regs: int
    start_pc: int


class ProgramBuilder:
    """Small mutable helper -- appends one node at a time, returns its
    index, lets Python-level code wire jump targets by index (exactly the
    role a `Branch`/`Loop`'s generated `if`/`while` source plays, but as
    data instead of text)."""

    def __init__(self, n_regs: int):
        self.n_regs = n_regs
        self.rows: list[tuple] = []  # (op, step_idx, a0, a1, a2, dest, next_, alt)

    def add(self, op, step_idx=NO_ARG, arg0=NO_ARG, arg1=NO_ARG, arg2=NO_ARG,
            dest=NO_ARG, next_=NO_ARG, alt=NO_ARG) -> int:
        self.rows.append((op, step_idx, arg0, arg1, arg2, dest, next_, alt))
        return len(self.rows) - 1

    def set_next(self, pc: int, next_: "int | None" = None, alt: "int | None" = None) -> None:
        """Patch a jump target after the target node exists (forward
        references) -- the flat-array equivalent of a generated `while`'s
        label being known only once the loop body has been emitted."""
        op, step_idx, a0, a1, a2, dest, cur_next, cur_alt = self.rows[pc]
        if next_ is not None:
            cur_next = next_
        if alt is not None:
            cur_alt = alt
        self.rows[pc] = (op, step_idx, a0, a1, a2, dest, cur_next, cur_alt)

    def build(self, start_pc: int) -> Program:
        arr = np.array(self.rows, dtype=np.int32)
        return Program(
            op=arr[:, 0].copy(), step_idx=arr[:, 1].copy(),
            arg0=arr[:, 2].copy(), arg1=arr[:, 3].copy(), arg2=arr[:, 4].copy(),
            dest=arr[:, 5].copy(), next_=arr[:, 6].copy(), alt=arr[:, 7].copy(),
            n_regs=self.n_regs, start_pc=start_pc,
        )


def make_ptr_table(cfuncs: "list") -> np.ndarray:
    return np.array([c.address for c in cfuncs], dtype=np.uint64)


# ---------------------------------------------------------------------------
# THE generic kernel. Compiled ONCE, ever, regardless of how many distinct
# programs (trees, Branch/Loop shapes, nesting depths) are run through it --
# this is the whole claim under test. `param_template` is a per-kernel
# (not per-row) float64 array already carrying every param() value at its
# designated register slot; each row starts from a fresh copy of it.
# ---------------------------------------------------------------------------
@njit(cache=True)
def run_program(op, step_idx, arg0, arg1, arg2, dest, next_, alt, start_pc,
                 ptr_table, param_template, row_regs, n_rows, out):
    """`row_regs[r, k]` is register `k`'s ROW-SPECIFIC initial value (a
    leaf/carry initial value) for row `r`, or NaN where the program's own
    `param_template` value should be kept untouched -- callers build
    `row_regs` with `np.nan` in every slot except the ones that are genuine
    per-row inputs, so a plain `if not isnan` decides which wins without a
    second bitmap array.

    `cache=True` deliberately -- item 5's decisive question. `ptr_table` is
    an ORDINARY ARGUMENT (a `uint64[:]`), never a module-level global and
    never a ctypes object numba specialises the compiled body on. Nothing
    about this function's own compiled code depends on which addresses
    happen to be in that array at any given call, so -- unlike §V's
    ctypes-global mechanism, which bakes the callee's address into the
    KERNEL's own specialisation and trips numba's "Cannot cache compiled
    function... dynamic globals" -- there is no dynamic global here at all
    to void the cache. `cache_check.py` confirms this empirically against a
    persistent build dir, across a real process restart."""
    n_regs = param_template.shape[0]
    for r in range(n_rows):
        regs = param_template.copy()
        for k in range(n_regs):
            v = row_regs[r, k]
            if v == v:  # not NaN
                regs[k] = v
        pc = start_pc
        while True:
            o = op[pc]
            if o == STEP1:
                addr = ptr_table[step_idx[pc]]
                a = regs[arg0[pc]]
                regs[dest[pc]] = call_f64_f64(addr, a)
                pc = next_[pc]
            elif o == STEP3:
                addr = ptr_table[step_idx[pc]]
                a = regs[arg0[pc]]
                b = regs[arg1[pc]]
                c = regs[arg2[pc]]
                regs[dest[pc]] = call_f64_3args(addr, a, b, c)
                pc = next_[pc]
            elif o == COND1:
                addr = ptr_table[step_idx[pc]]
                a = regs[arg0[pc]]
                cond = call_bool_1arg(addr, a)
                pc = next_[pc] if cond else alt[pc]
            elif o == COND2:
                addr = ptr_table[step_idx[pc]]
                a = regs[arg0[pc]]
                b = regs[arg1[pc]]
                cond = call_bool_2args(addr, a, b)
                pc = next_[pc] if cond else alt[pc]
            elif o == COND3:
                addr = ptr_table[step_idx[pc]]
                a = regs[arg0[pc]]
                b = regs[arg1[pc]]
                c = regs[arg2[pc]]
                cond = call_bool_3args(addr, a, b, c)
                pc = next_[pc] if cond else alt[pc]
            elif o == SET_ZERO:
                regs[dest[pc]] = 0.0
                pc = next_[pc]
            elif o == INCR:
                regs[dest[pc]] += 1.0
                pc = next_[pc]
            elif o == JUMP:
                pc = next_[pc]
            else:  # LEAF
                out[r] = regs[dest[pc]]
                break
