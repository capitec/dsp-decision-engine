"""`@cfunc` equivalents of decider2's own
`decider2.examples.playground_control_flow` section-3 steps (`is_high_income`,
`term_still_short`, `fast_track_should_continue`, `fast_bump`, `slow_bump`) --
same arithmetic, same order of operations, so the interpreter's answers
should match decider2's real codegen exactly (`compare_vs_codegen.py` asserts
this with `np.array_equal`, not `np.isclose`).

Every threshold that playground_control_flow declares via `param()` (floor,
micro_steps, jump, step_size) is a plain float64 ARGUMENT here too, never a
baked-in literal -- matching doc 08 §2.1 ("a rule's thresholds are arguments,
not emitted literals") and setting up item 6 (`params_check.py`) for free:
retuning is changing a value passed in, not touching this file.
"""
from __future__ import annotations

from numba import cfunc, types

f64 = types.float64
b1 = types.boolean


@cfunc(b1(f64), cache=True)
def is_high_income(high_income):
    """`playground_control_flow.is_high_income`: `return high_income`.
    `high_income` crosses as 0.0/1.0 float64 (this experiment's register
    file is uniformly float64 -- see `interpreter.py`'s module docstring)."""
    return high_income != 0.0


@cfunc(b1(f64, f64, f64), cache=True)
def term_still_short(term_cap, loop_idx, floor):
    """`playground_control_flow.term_still_short`:
    `return term_cap < floor and loop_idx < 1000`."""
    return term_cap < floor and loop_idx < 1000.0


@cfunc(b1(f64, f64), cache=True)
def fast_track_should_continue(loop_idx, micro_steps):
    """`playground_control_flow.fast_track_should_continue`:
    `return loop_idx < micro_steps`."""
    return loop_idx < micro_steps


@cfunc(f64(f64, f64, f64), cache=True)
def fast_bump(term_cap, loop_idx, jump):
    """`playground_control_flow.fast_bump`:
    `return term_cap + jump + 0.0 * loop_idx`."""
    return term_cap + jump + 0.0 * loop_idx


@cfunc(f64(f64, f64, f64), cache=True)
def slow_bump(term_cap, loop_idx, step_size):
    """`playground_control_flow.slow_bump`:
    `return term_cap + step_size + 0.0 * loop_idx`."""
    return term_cap + step_size + 0.0 * loop_idx
