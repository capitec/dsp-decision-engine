"""The generic Branch/Loop interpreter (doc 03 §8.2/§8.3) — replaces
`branch.py`/`loop.py`'s former per-construct emitted `if`/`elif`/`while`
source with ONE shared `@njit(cache=True)` walker, `run_program`, driven by
a DATA program.

**Started from `experimentation/cfunc-pointer-interpreter/` (EXPERIMENTS.md
§W and its correction), then corrected once more.** That experiment's own
mechanism — a `@numba.cfunc` address read from a runtime-indexed `uint64`
array — is exactly what `decider2.trees.interpreter` also validated for
trees, and §W's own report states the rule precisely: "a pointer passed as
an ARGUMENT caches; a symbol captured as a compile-time GLOBAL does not."
Building THIS module against that rule surfaced a sharper version of it
that neither §V nor §W's own harness could see, because neither drove the
result through `decider2.compile.driver._try_njit`: **decider2 always
wraps a `Step.fn` in its OWN, separate `njit(cache=True)` — one this
module's own wrapper functions cannot avoid (it is not a Branch/Loop
policy choice, it is the fixed calling convention doc 03 §1 gives every
step) — and a pointer table built at import time and merely REFERENCED
from inside THAT wrapping (even only to hand it on to `run_program` as an
argument one level down) is enough to reproduce §V's "cannot cache"
failure, except SILENTLY: no warning, a real .nbi/.nbc pair written and
reloaded, and a segfault on the warm run** (empirically confirmed: cold
process compiles and runs correctly; every subsequent process loading the
same `.nbi`/`.nbc` pair crashes, because the addresses baked into that
cached compile belong to the COLD process's memory, not the one now
reading them). Passing the same table one level further down — as `run_
program`'s own argument, exactly as the lifted experiment does — does not
help, because the Step-level wrapper still has to *read the table from
somewhere* to hand it down, and reading it from a module global is the
part that breaks, regardless of how many frames later the address is
actually used.

**The fix: no address ever crosses a `cache=True` boundary as data.**
`run_program` still walks one shared, generic, data-encoded program — the
control-flow OPCODES below (`STEP`/`COND`/`CMP_LIT`/`SET_ZERO`/`SET_LIT`/
`INCR`/`JUMP`/`LEAF`) are exactly the closed, inline-branched vocabulary
EXPERIMENTS.md §Q/§W and `decider2.trees.interpreter` already establish —
but a `STEP`/`COND` node's callee is resolved by calling `call_step`/
`call_cond`, two ordinary `@njit` functions passed into `run_program` as
ARGUMENTS, each holding one Branch/Loop construct's own small, closed
`step_idx` switch over DIRECT, BY-NAME calls to that construct's condition/
arm/body steps (`_engine.render_step_thunk`, `branch.py`/`loop.py`'s
`_register_thunk`) — the exact calling convention `decider2.compile.
codegen`/`driver` already use successfully for a fused kernel's own steps,
proven safe by 531 pre-existing tests before this migration touched it.
Passing an already-`@njit`-compiled FUNCTION as an argument to another
`@njit(cache=True)` function is an ordinary, fully-supported numba
mechanism (confirmed safe cold *and* warm, repeatedly, as part of this
finding) — categorically different from passing a raw integer address read
from a global, which is what actually broke.

The result is still exactly what doc 03 §8.2/§8.3 asks for and what this
stage's brief asks for: no per-construct SOURCE TEXT for the ROUTING (the
loop head-check, the back-edge, which arm's result to return) — that stays
one shared kernel, walked from a DATA program, same as a tree's own
`if`/`elif` retirement. What is NOT achieved, and is reported as a
correction to the brief rather than papered over, is calling an arm/body
step through a raw C-ABI pointer read from a table; `call_step`/`call_cond`
resolve it through a small, per-construct, by-name switch instead — bounded
by that construct's own step count, never by nesting depth or iteration
count, and therefore never the source of the problem this migration exists
to retire.

**Registers.** Every value a program touches — a carry, `loop_idx`, a leaf
input, a `param()` value — lives in one fixed-size per-call float64 `regs`
array, at an index decided once when the program is built
(`control_flow._engine`'s encoder). `regs` is fully built by the caller
from its own arguments before `run_program` is ever called; there is
nothing here for doc 08 §2's "retuning a param never recompiles" guarantee
to trip on, because a retuned value only ever changes what `regs` holds,
never what any `@njit` specialisation compiles against.

**`CMP_LIT`.** The one opcode with no by-name call at all: `regs[arg0] ==
lit[pc]` (or any of the tree interpreter's six comparisons, reused —
`decider2.trees.interpreter.compare` — rather than a second copy), used
only for Branch's routing form: "which arm produced this modifies value"
is an ordinary integer-against-literal test, not a call to arbitrary code,
so it gets the tree's array-compare treatment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from decider2.trees.interpreter import compare

__all__ = [
    "STEP", "COND", "CMP_LIT", "SET_ZERO", "SET_LIT", "INCR", "JUMP", "LEAF",
    "CALL_ARITY", "NO_ARG",
    "Program", "ProgramBuilder",
    "run_program",
]

# --- opcodes -----------------------------------------------------------
STEP = 0
COND = 1
CMP_LIT = 2
SET_ZERO = 3
SET_LIT = 4
INCR = 5
JUMP = 6
LEAF = 7

NO_ARG = -1

# The fixed width of `call_step`/`call_cond`'s own argument list — every
# construct's switch ignores whichever trailing slots its own steps don't
# need (`_engine.call_arg_regs` pads with register 0, never read for an
# unused slot because the underlying step's OWN by-name call only takes as
# many positional arguments as it declares). 6 is generous headroom over
# what nesting needs in this build's own test suite
# (`test_nested_loop_of_branch_assert_equivalent` needs 5) — a real,
# reported scope cut past which `_engine.call_arg_regs` raises a clear,
# named build error rather than silently truncating an argument, the same
# shape `_engine.single_step`'s "exactly one step" limit already is.
CALL_ARITY = 6


@dataclass(frozen=True)
class Program:
    """One Branch/Loop construct's control-flow program, entirely as flat
    int32 arrays — what `branch.py`/`loop.py` used to instead emit as a
    `.py` file's `if`/`elif`/`while` source. `n_regs` sizes the per-call
    register scratch; `start_pc` is where the walk begins. `step_idx[pc]`
    is this construct's OWN index for whichever step a `STEP`/`COND` node
    calls — resolved by that construct's own `call_step`/`call_cond`
    (`_engine.render_step_thunk`), never by a shared table."""

    op: np.ndarray
    step_idx: np.ndarray
    arg0: np.ndarray
    arg1: np.ndarray
    arg2: np.ndarray
    arg3: np.ndarray
    arg4: np.ndarray
    arg5: np.ndarray
    lit: np.ndarray
    dest: np.ndarray
    next_: np.ndarray
    alt: np.ndarray
    n_regs: int
    start_pc: int


class ProgramBuilder:
    """Small mutable helper — appends one node at a time, returns its
    index, lets `_engine.py`'s encoder wire jump targets by index (exactly
    the role a generated `if`/`while`'s own source text used to play, but
    as data instead of text)."""

    def __init__(self, n_regs: int):
        self.n_regs = n_regs
        self.rows: list[tuple] = []

    def add(
        self, op, step_idx=NO_ARG, arg0=NO_ARG, arg1=NO_ARG, arg2=NO_ARG, arg3=NO_ARG,
        arg4=NO_ARG, arg5=NO_ARG, lit=0.0, dest=NO_ARG, next_=NO_ARG, alt=NO_ARG,
    ) -> int:
        self.rows.append(
            (op, step_idx, arg0, arg1, arg2, arg3, arg4, arg5, lit, dest, next_, alt)
        )
        return len(self.rows) - 1

    def set_next(self, pc: int, next_: "int | None" = None, alt: "int | None" = None) -> None:
        row = list(self.rows[pc])
        if next_ is not None:
            row[10] = next_
        if alt is not None:
            row[11] = alt
        self.rows[pc] = tuple(row)

    def build(self, start_pc: int) -> Program:
        int_cols = np.array(
            [r[:8] + r[9:] for r in self.rows], dtype=np.int32
        )  # everything except `lit` (index 8)
        lit_col = np.array([r[8] for r in self.rows], dtype=np.float64)
        return Program(
            op=int_cols[:, 0].copy(), step_idx=int_cols[:, 1].copy(),
            arg0=int_cols[:, 2].copy(), arg1=int_cols[:, 3].copy(),
            arg2=int_cols[:, 4].copy(), arg3=int_cols[:, 5].copy(),
            arg4=int_cols[:, 6].copy(), arg5=int_cols[:, 7].copy(),
            lit=lit_col,
            dest=int_cols[:, 8].copy(), next_=int_cols[:, 9].copy(), alt=int_cols[:, 10].copy(),
            n_regs=self.n_regs, start_pc=start_pc,
        )


# ---------------------------------------------------------------------------
# THE generic kernel. ONE Python-level definition, ever, regardless of how
# many distinct Branch/Loop programs are run through it; numba specialises
# it per distinct `(call_step, call_cond)` function pair the same way it
# specialises any generic `@njit` function per argument type — this is
# ordinary, well-supported numba behaviour (confirmed safe cold and warm
# by this stage's own cache probe), unlike a raw pointer read from a
# global, which is not. `regs` is this ONE call's fully-populated register
# array, built by the caller, mutated in place as the walk runs.
#
# Deliberately NOT `cache=True`. Doc 05 §4.2's cache is keyed per COMPILED
# FUNCTION, not per specialisation — and this function gets a genuinely new
# specialisation per distinct `(call_step, call_cond)` pair, i.e. per
# distinct Branch/Loop CONSTRUCT that ever runs in this interpreter
# process, sharing ONE on-disk index file across all of them. Measured
# empirically while building this stage: that index accumulates one entry
# per construct forever (nothing here ever evicts it), and loading it back
# requires deserialising every PRIOR entry's own dispatcher-type reference —
# including ones pointing at a NOW-regenerated-away, content-addressed
# generated module from an earlier build. That produced a real, reproduced
# `ReferenceError: underlying object has vanished` deep inside numba's own
# cache loader, not a decider2 bug to fix here. `run_program`'s own body is
# a handful of comparisons over a small, fixed opcode set — recompiling it
# once per process per distinct construct (never per row, never on a
# retune, doc 08 §2's guarantee is untouched) costs microseconds, which is
# the honest trade against an index that corrupts itself over the lifetime
# of a long-lived build directory. The WRAPPER functions this module's
# callers generate (`{name}_path`, `{name}_{target}`, ...) keep `cache=
# True` and cache normally — this is the one function in the whole
# migration where that flag is deliberately absent, and this comment is
# why.
# ---------------------------------------------------------------------------
@njit
def run_program(
    op, step_idx, arg0, arg1, arg2, arg3, arg4, arg5, lit, dest, next_, alt, start_pc,
    call_step, call_cond, regs,
):
    pc = start_pc
    while True:
        o = op[pc]
        if o == STEP:
            regs[dest[pc]] = call_step(
                step_idx[pc], regs[arg0[pc]], regs[arg1[pc]], regs[arg2[pc]],
                regs[arg3[pc]], regs[arg4[pc]], regs[arg5[pc]],
            )
            pc = next_[pc]
        elif o == COND:
            cond = call_cond(
                step_idx[pc], regs[arg0[pc]], regs[arg1[pc]], regs[arg2[pc]],
                regs[arg3[pc]], regs[arg4[pc]], regs[arg5[pc]],
            )
            pc = next_[pc] if cond else alt[pc]
        elif o == CMP_LIT:
            cond = compare(arg1[pc], regs[arg0[pc]], lit[pc])
            pc = next_[pc] if cond else alt[pc]
        elif o == SET_ZERO:
            regs[dest[pc]] = 0.0
            pc = next_[pc]
        elif o == SET_LIT:
            regs[dest[pc]] = lit[pc]
            pc = next_[pc]
        elif o == INCR:
            regs[dest[pc]] += 1.0
            pc = next_[pc]
        elif o == JUMP:
            pc = next_[pc]
        else:  # LEAF
            return regs[dest[pc]]
