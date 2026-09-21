"""Control for the caching-cost side-measurement: the SAME 3-step shape as
`pipeline_module.py`, but `tree_decision` is a self-contained numeric
function with NO global array and NO ctypes reference at all — genuinely
numba-cacheable, unlike a version that reads module-level tree arrays
(numba's own "dynamic globals" warning covers `large global arrays` too,
not only ctypes pointers, so a global-array version would not isolate
the ctypes-specific cost this control exists to isolate).

Isolates whether numba's on-disk cache persists ACROSS PROCESSES for an
ordinary fused kernel, as a baseline for `pipeline_module`'s measured
cold-every-time cost (see RESULTS.md's caching-cost note).
"""
from __future__ import annotations


def scale_features(f0: float, f1: float) -> float:
    return f0 / 100.0 - f1 / 100.0


def tree_decision(f0: float, f1: float, f2: float, f3: float) -> float:
    """A hand-written 4-leaf decision, same arity as `pipeline_module.
    tree_decision`, zero external state.

    Implements: cabi-fusion-check-control
    """
    eps = f3 * 0.0  # touches f3 without changing the decision — kept at the
    # same 4-feature arity as pipeline_module.tree_decision for a fair
    # side-by-side, per decider2's "every declared input must be read" rule
    if f0 < 50.0:
        if f1 < 50.0:
            return 1.0 + eps
        return 2.0 + eps
    else:
        if f2 < 50.0:
            return 3.0 + eps
        return 4.0 + eps


def apply_bonus(tree_decision: float, scale_features: float) -> float:
    return tree_decision + 0.01 * scale_features
