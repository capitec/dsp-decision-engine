"""A REAL, importable module (not a closure, not exec'd) holding three
ordinary decider2 steps, the middle one making a per-row C-ABI call into
`rust_cabi_tree`. It has to be a real file on disk: `decider2.compile.
codegen.emit_kernel_source` generates `from {step.fn.__module__} import
{step.fn.__name__}` for every step in a fused group (doc 05 §4.1), and
`decider2.compile.driver.build_driver`'s own `ImportError` handling
(driver.py, "kernel source could not import") is what a step defined
inside a function or a REPL session hits instead — this file exists so the
fusion check is exercised under the SAME constraint any real step is.

Tree data (`_KIND`/.../`_LEAF`) and the C-ABI function pointer
(`_walk_row`) are module-level globals, referenced directly from inside
`tree_decision`'s body — the same pattern the original C-stand-in probe
(`/tmp/claude-1000/cabi/tree.py`) used: numba treats a module-level global
read inside `@njit` as a compile-time constant, so no plumbing of tree-array
pointers through decider2's `param()` machinery is needed for this check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "tree-codegen-vs-interpreted"))

import tree_shapes as ts  # noqa: E402
import tree_walk_cabi as tw  # noqa: E402

# A small, real tree (not a toy 1-node stand-in) — full-binary depth 3 over
# 4 numeric features, built with the SAME `tree_shapes.build_full_binary`
# every other measurement in this experiment uses.
_SHAPE = ts.build_full_binary(depth=3, n_features=4, seed=11)
_FLAT = ts.to_flat_tree(_SHAPE)
_KIND, _FEAT, _OP, _THRESH = _FLAT.kind, _FLAT.feat_idx, _FLAT.op_code, _FLAT.thresh
_PAT_START, _PAT_COUNT, _PATTERNS = _FLAT.pat_start, _FLAT.pat_count, _FLAT.patterns
_LEFT, _RIGHT, _LEAF = _FLAT.left, _FLAT.right, _FLAT.leaf_value
_STR_ROW = np.zeros(0, dtype=np.int32)  # this tree has no string tests

# `walk_row_cabi` is the SAME njit wrapper `bench_perf.py` times — reused,
# not re-implemented, so "does this stay fused" is asked about the exact
# call the performance numbers are about.
_walk_row_cabi = tw.walk_row_cabi


def scale_features(f0: float, f1: float) -> float:
    """Step 1 — an ordinary numeric step, nothing to do with the tree.
    Proves the group *around* the C-ABI step is itself fusable, not just
    the one step containing the call."""
    return f0 / 100.0 - f1 / 100.0


def tree_decision(f0: float, f1: float, f2: float, f3: float) -> float:
    """Step 2 — the tree walk, via ONE `extern "C"` call into
    `rust_cabi_tree::walk_row` per row, from inside what will become this
    fused kernel's row loop.

    Implements: cabi-fusion-check
    """
    row = np.empty(4)
    row[0] = f0
    row[1] = f1
    row[2] = f2
    row[3] = f3
    return _walk_row_cabi(_KIND, _FEAT, _OP, _THRESH, _PAT_START, _PAT_COUNT,
                           _PATTERNS, _LEFT, _RIGHT, _LEAF, row, _STR_ROW)


def apply_bonus(tree_decision: float, scale_features: float) -> float:
    """Step 3 — reads BOTH an earlier step's output by name (ordinary
    intra-group wiring, doc 05 §7) and the C-ABI step's own output,
    proving neither kind of cross-step data flow forces a split around
    the call in between."""
    return tree_decision + 0.01 * scale_features
