"""Item 2(d) + item 3's shared program: the flat-array encoding of
`Loop(Branch(steps1, Loop(steps2)))` -- decider2's own
`playground_control_flow.SearchForViableTerm` (outer `Loop`, carries
`term_cap`) wrapping `AdjustmentStrategy` (`Branch` on `is_high_income`)
whose True arm is `FastTrackBump` (inner `Loop`) and False arm is
`slow_bump` (a plain step).

Register layout (n_regs=8), fixed for the life of this program:

  0  term_cap        carry -- row input, then mutated every body call
  1  outer_idx       outer Loop's loop_idx -- SET_ZERO on entry, INCR on
                      the outer back-edge
  2  inner_idx       inner Loop's loop_idx -- SET_ZERO EVERY time the
                      Branch's True arm is entered (a fresh loop each outer
                      iteration), INCR on the inner back-edge
  3  high_income     row input (0.0/1.0), read-only
  4  floor           param -- term_still_short's threshold
  5  micro_steps     param -- fast_track_should_continue's threshold
  6  jump            param -- fast_bump's step size
  7  step_size       param -- slow_bump's step size

Node graph (12 nodes, `start_pc=0`) -- see `interpreter.py`'s module
docstring for the opcode semantics:

  0  SET_ZERO  outer_idx=0                          -> 1
  1  COND3     term_still_short(term_cap,outer_idx,floor)   T->2  F->11 (LEAF)
  2  COND1     is_high_income(high_income)                  T->3  F->8
  3  SET_ZERO  inner_idx=0                           -> 4
  4  COND2     fast_track_should_continue(inner_idx,micro_steps) T->5 F->7
  5  STEP3     fast_bump(term_cap,inner_idx,jump) -> term_cap    -> 6
  6  INCR      inner_idx += 1                        -> 4   (inner back-edge)
  7  JUMP      -> 10                                          (FastTrack exit)
  8  STEP3     slow_bump(term_cap,outer_idx,step_size) -> term_cap -> 9
  9  JUMP      -> 10
  10 INCR      outer_idx += 1                        -> 1   (outer back-edge)
  11 LEAF      out[row] = term_cap
"""
from __future__ import annotations

import numpy as np

from interpreter import (
    COND1, COND2, COND3, INCR, JUMP, LEAF, SET_ZERO, STEP3,
    ProgramBuilder, make_ptr_table,
)
import steps_nested as sn

REG_TERM_CAP = 0
REG_OUTER_IDX = 1
REG_INNER_IDX = 2
REG_HIGH_INCOME = 3
REG_FLOOR = 4
REG_MICRO_STEPS = 5
REG_JUMP = 6
REG_STEP_SIZE = 7
N_REGS = 8

STEP_IS_HIGH_INCOME = 0
STEP_TERM_STILL_SHORT = 1
STEP_FAST_TRACK_SHOULD_CONTINUE = 2
STEP_FAST_BUMP = 3
STEP_SLOW_BUMP = 4

PTR_TABLE = make_ptr_table([
    sn.is_high_income, sn.term_still_short, sn.fast_track_should_continue,
    sn.fast_bump, sn.slow_bump,
])

# Playground's own defaults (matches SearchForViableTermParams' pydantic
# field defaults exactly -- see the probe in this agent's build notes).
DEFAULT_PARAMS = {
    REG_FLOOR: 40.0,
    REG_MICRO_STEPS: 3.0,
    REG_JUMP: 4.0,
    REG_STEP_SIZE: 3.0,
}


def build_nested_program():
    b = ProgramBuilder(n_regs=N_REGS)
    n0 = b.add(SET_ZERO, dest=REG_OUTER_IDX)
    n1 = b.add(COND3, step_idx=STEP_TERM_STILL_SHORT,
               arg0=REG_TERM_CAP, arg1=REG_OUTER_IDX, arg2=REG_FLOOR)
    n2 = b.add(COND1, step_idx=STEP_IS_HIGH_INCOME, arg0=REG_HIGH_INCOME)
    n3 = b.add(SET_ZERO, dest=REG_INNER_IDX)
    n4 = b.add(COND2, step_idx=STEP_FAST_TRACK_SHOULD_CONTINUE,
               arg0=REG_INNER_IDX, arg1=REG_MICRO_STEPS)
    n5 = b.add(STEP3, step_idx=STEP_FAST_BUMP,
               arg0=REG_TERM_CAP, arg1=REG_INNER_IDX, arg2=REG_JUMP, dest=REG_TERM_CAP)
    n6 = b.add(INCR, dest=REG_INNER_IDX)
    n7 = b.add(JUMP)
    n8 = b.add(STEP3, step_idx=STEP_SLOW_BUMP,
               arg0=REG_TERM_CAP, arg1=REG_OUTER_IDX, arg2=REG_STEP_SIZE, dest=REG_TERM_CAP)
    n9 = b.add(JUMP)
    n10 = b.add(INCR, dest=REG_OUTER_IDX)
    n11 = b.add(LEAF, dest=REG_TERM_CAP)

    b.set_next(n0, next_=n1)
    b.set_next(n1, next_=n2, alt=n11)
    b.set_next(n2, next_=n3, alt=n8)
    b.set_next(n3, next_=n4)
    b.set_next(n4, next_=n5, alt=n7)
    b.set_next(n5, next_=n6)
    b.set_next(n6, next_=n4)
    b.set_next(n7, next_=n10)
    b.set_next(n8, next_=n9)
    b.set_next(n9, next_=n10)
    b.set_next(n10, next_=n1)

    return b.build(start_pc=n0)


def param_template(overrides: "dict | None" = None) -> np.ndarray:
    values = dict(DEFAULT_PARAMS)
    if overrides:
        values.update(overrides)
    t = np.zeros(N_REGS, dtype=np.float64)
    for reg, v in values.items():
        t[reg] = v
    return t


def make_row_regs(high_income: np.ndarray, term_cap0: np.ndarray) -> np.ndarray:
    n = high_income.shape[0]
    row_regs = np.full((n, N_REGS), np.nan)
    row_regs[:, REG_TERM_CAP] = term_cap0
    row_regs[:, REG_HIGH_INCOME] = high_income.astype(np.float64)
    return row_regs
