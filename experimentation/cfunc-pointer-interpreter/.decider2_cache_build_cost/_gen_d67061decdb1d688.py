"""Generated kernel for group 'g0_term_cap'.

Content-addressed: this file's name is a hash of its own bytes
(decider2.compile.cache), so an edit here always produces a new
file rather than a stale cache hit at either the numba or the
CPython .pyc layer (doc 05 §4.2). Do not hand-edit — regenerate
from the pipeline instead.
"""
from __future__ import annotations

from numba import njit, prange

from decider2_generated._gen_ec43954ff89a85e4 import search_for_viable_term_term_cap as _src_term_cap

_compiled_term_cap = njit(cache=True)(_src_term_cap)

@njit(cache=True)
def kernel(arr_high_income, arr_term_cap, p_term_cap_term_still_short__floor, p_term_cap_adjustment_strategy__fast_track_bump__fast_track_should_continue__micro_steps, p_term_cap_adjustment_strategy__fast_track_bump__fast_bump__jump, p_term_cap_adjustment_strategy__slow_bump__step_size, out_term_cap):
    n = arr_high_income.shape[0]
    for i in range(n):
        v_high_income = arr_high_income[i]
        v_term_cap = arr_term_cap[i]
        v_term_cap = _compiled_term_cap(v_high_income, v_term_cap, p_term_cap_term_still_short__floor, p_term_cap_adjustment_strategy__fast_track_bump__fast_track_should_continue__micro_steps, p_term_cap_adjustment_strategy__fast_track_bump__fast_bump__jump, p_term_cap_adjustment_strategy__slow_bump__step_size)
        out_term_cap[i] = v_term_cap
