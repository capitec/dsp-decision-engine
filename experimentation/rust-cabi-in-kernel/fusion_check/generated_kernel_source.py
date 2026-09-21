"""Generated kernel for group 'g0_scale_features_tree_decision_apply_bonus'.

Content-addressed: this file's name is a hash of its own bytes
(decider2.compile.cache), so an edit here always produces a new
file rather than a stale cache hit at either the numba or the
CPython .pyc layer (doc 05 §4.2). Do not hand-edit — regenerate
from the pipeline instead.
"""
from __future__ import annotations

from numba import njit, prange

from pipeline_module import scale_features as _src_scale_features
from pipeline_module import tree_decision as _src_tree_decision
from pipeline_module import apply_bonus as _src_apply_bonus

_compiled_scale_features = njit(cache=True)(_src_scale_features)
_compiled_tree_decision = njit(cache=True)(_src_tree_decision)
_compiled_apply_bonus = njit(cache=True)(_src_apply_bonus)

@njit(cache=True)
def kernel(arr_f0, arr_f1, arr_f2, arr_f3, out_apply_bonus):
    n = arr_f0.shape[0]
    for i in range(n):
        v_f0 = arr_f0[i]
        v_f1 = arr_f1[i]
        v_f2 = arr_f2[i]
        v_f3 = arr_f3[i]
        v_scale_features = _compiled_scale_features(v_f0, v_f1)
        v_tree_decision = _compiled_tree_decision(v_f0, v_f1, v_f2, v_f3)
        v_apply_bonus = _compiled_apply_bonus(v_tree_decision, v_scale_features)
        out_apply_bonus[i] = v_apply_bonus
