"""Item 2's coverage requirement: the ONE generic kernel in `interpreter.py`
(`run_program`, compiled exactly once, reused unchanged below) correctly
runs three of the four required shapes, each checked against an independent
pure-Python reference over random per-row data. The fourth shape (nested
`Loop(Branch(steps1, Loop(steps2)))`) is `nested_program.py`, checked
against decider2's own REAL `Branch`/`Loop` codegen instead of a Python
reference -- the stronger check, and the one item 3 also uses.

Run: <repo>/.venv/bin/python experimentation/cfunc-pointer-interpreter/test_shapes.py
"""
from __future__ import annotations

import numpy as np
from numba import cfunc, types

from interpreter import (
    COND1, INCR, JUMP, LEAF, SET_ZERO, STEP1,
    ProgramBuilder, make_ptr_table, run_program,
)

RNG = np.random.default_rng(20260921)


def run(prog, ptr_table, row_regs, n_rows) -> np.ndarray:
    out = np.empty(n_rows, dtype=np.float64)
    param_template = np.zeros(prog.n_regs, dtype=np.float64)  # no params in these 3 shapes
    run_program(
        prog.op, prog.step_idx, prog.arg0, prog.arg1, prog.arg2, prog.dest,
        prog.next_, prog.alt, prog.start_pc, ptr_table, param_template,
        row_regs, n_rows, out,
    )
    return out


# ---------------------------------------------------------------------------
# (a) straight-line sequence: out = (x + 1) * 2 - 3, three chained STEP1s.
# ---------------------------------------------------------------------------
def test_straight_line() -> None:
    @cfunc(types.float64(types.float64))
    def add1(x):
        return x + 1.0

    @cfunc(types.float64(types.float64))
    def mul2(x):
        return x * 2.0

    @cfunc(types.float64(types.float64))
    def sub3(x):
        return x - 3.0

    ptr_table = make_ptr_table([add1, mul2, sub3])
    b = ProgramBuilder(n_regs=2)  # reg0 = value, reg1 = scratch
    n0 = b.add(STEP1, step_idx=0, arg0=0, dest=1)
    n1 = b.add(STEP1, step_idx=1, arg0=1, dest=1)
    n2 = b.add(STEP1, step_idx=2, arg0=1, dest=1)
    n3 = b.add(LEAF, dest=1)
    b.set_next(n0, next_=n1)
    b.set_next(n1, next_=n2)
    b.set_next(n2, next_=n3)
    prog = b.build(start_pc=n0)

    n_rows = 10_000
    x = RNG.uniform(-100, 100, n_rows)
    row_regs = np.full((n_rows, 2), np.nan)
    row_regs[:, 0] = x
    got = run(prog, ptr_table, row_regs, n_rows)
    expect = (x + 1.0) * 2.0 - 3.0
    assert np.array_equal(got, expect), "straight-line mismatch"
    print(f"(a) straight-line sequence: OK ({n_rows} rows, exact match)")


# ---------------------------------------------------------------------------
# (b) Branch: out = x + 10 if x > 50 else x - 5.
# ---------------------------------------------------------------------------
def test_branch() -> None:
    @cfunc(types.boolean(types.float64))
    def is_high(x):
        return x > 50.0

    @cfunc(types.float64(types.float64))
    def bump_high(x):
        return x + 10.0

    @cfunc(types.float64(types.float64))
    def bump_low(x):
        return x - 5.0

    ptr_table = make_ptr_table([is_high, bump_high, bump_low])
    b = ProgramBuilder(n_regs=1)
    n_cond = b.add(COND1, step_idx=0, arg0=0)
    n_true = b.add(STEP1, step_idx=1, arg0=0, dest=0)
    n_false = b.add(STEP1, step_idx=2, arg0=0, dest=0)
    n_leaf = b.add(LEAF, dest=0)
    b.set_next(n_cond, next_=n_true, alt=n_false)
    b.set_next(n_true, next_=n_leaf)
    b.set_next(n_false, next_=n_leaf)
    prog = b.build(start_pc=n_cond)

    n_rows = 10_000
    x = RNG.uniform(0, 100, n_rows)
    row_regs = np.full((n_rows, 1), np.nan)
    row_regs[:, 0] = x
    got = run(prog, ptr_table, row_regs, n_rows)
    expect = np.where(x > 50.0, x + 10.0, x - 5.0)
    assert np.array_equal(got, expect), "branch mismatch"
    print(f"(b) Branch: OK ({n_rows} rows, exact match)")


# ---------------------------------------------------------------------------
# (c) Loop with a REAL, data-dependent early exit: extend `term_cap` by 1.0
# each iteration while `principal / term_cap > ceiling`, bounded at 1000 --
# same shape as decider2's own `ExtendTermToFit` (playground_control_flow.py
# §2), reimplemented as flat-array data instead of generated source.
# ---------------------------------------------------------------------------
def test_loop_early_exit() -> None:
    @cfunc(types.boolean(types.float64, types.float64, types.float64))
    def instalment_fits(term_cap, principal, ceiling):
        return not ((principal / term_cap) > ceiling)  # true = STOP looping

    @cfunc(types.float64(types.float64))
    def add_one(term_cap):
        return term_cap + 1.0

    ptr_table = make_ptr_table([instalment_fits, add_one])
    # regs: 0=term_cap (carry), 1=principal (row input), 2=ceiling (param)
    from interpreter import COND3
    b = ProgramBuilder(n_regs=3)
    n_check = b.add(COND3, step_idx=0, arg0=0, arg1=1, arg2=2)
    n_body = b.add(STEP1, step_idx=1, arg0=0, dest=0)
    n_leaf = b.add(LEAF, dest=0)
    # cond TRUE ("fits", i.e. stop) -> leaf; FALSE (doesn't fit) -> body -> back to check
    b.set_next(n_check, next_=n_leaf, alt=n_body)
    b.set_next(n_body, next_=n_check)
    prog = b.build(start_pc=n_check)

    n_rows = 5_000
    principal = RNG.uniform(1_000, 200_000, n_rows)
    term_cap0 = RNG.uniform(6, 60, n_rows)
    ceiling = 2500.0
    row_regs = np.full((n_rows, 3), np.nan)
    row_regs[:, 0] = term_cap0
    row_regs[:, 1] = principal
    row_regs[:, 2] = ceiling
    got = run(prog, ptr_table, row_regs, n_rows)

    # pure-Python/numpy reference, no generic kernel involved
    expect = np.empty(n_rows)
    early_exit_count = 0
    ran_zero_iters = 0
    for i in range(n_rows):
        tc = term_cap0[i]
        it = 0
        while (principal[i] / tc) > ceiling and it < 1000:
            tc += 1.0
            it += 1
        expect[i] = tc
        if it < 1000:
            early_exit_count += 1
        if it == 0:
            ran_zero_iters += 1
    assert np.array_equal(got, expect), "loop-early-exit mismatch"
    print(
        f"(c) Loop with early exit: OK ({n_rows} rows, exact match; "
        f"{ran_zero_iters} rows exited on iteration 0, "
        f"{n_rows - early_exit_count} rows hit the 1000-iteration bound)"
    )


if __name__ == "__main__":
    test_straight_line()
    test_branch()
    test_loop_early_exit()
    print("\nAll three standalone shapes verified against an independent reference.")
