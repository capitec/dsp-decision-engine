"""Where should the kind discriminator live: a separate `feat_kind` int32
array, or the high bits of `feat_idx`?

Both are program data; the question is only whether one fewer array load
per node is worth a shift-and-mask. Measured on the SAME 40-leaf mixed
tree and 200k-row frame `bench_typed_features.py` uses, with the two
walkers otherwise identical and the whole chain inlined into the kernel
(the production shape). Appends to `measurements.jsonl` with
`metric: feat_kind_separate_ns_per_row` / `feat_kind_packed_ns_per_row`.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from numba import njit

import bench_typed_features as b
from decider2 import flow
from decider2.compile.driver import _fill_array, _typed_counts, _typed_input_arrays, _typed_params, numpy_dtype
from decider2.trees import tree_module
from decider2.trees.interpreter import BOOL, CMP, F64, I64, IS_TRUE, LEAF, compare

variant = "mixed"
tree, ft = b.build_tree(variant)
frame = b.build_frame(variant, 200_000)
built = tree_module(tree, feature_types=ft)
step = built.encoded.path_step
pipeline = flow(built.module)
by_name = {i.name: i for i in pipeline.interface.inputs}
registry = {
    inp.name: frame[inp.name].to_numpy().astype(numpy_dtype(by_name[inp.name].annotation), copy=False)
    for inp in step.inputs
}
arrays = _typed_input_arrays(step, registry)
thr_f, thr_i = _typed_params(step, tuple(d.default for d in step.params))
thr_f = thr_f or (0.0,)
thr_i = thr_i or (0,)
n = frame.height
n_f, n_i, n_b, n_c, n_s = _typed_counts(step)

enc = built.encoded.arrays
kind = np.array(enc["kind"], np.int32)
feat_kind = np.array(enc["feat_kind"], np.int32)
feat_idx = np.array(enc["feat_idx"], np.int32)
op = np.array(enc["op"], np.int32)
thr_slot = np.array(enc["thr_slot"], np.int32)
then_ = np.array(enc["then"], np.int32)
else_ = np.array(enc["else"], np.int32)
leaf_value = np.array(enc["leaf_value"], np.int64)
packed = np.array([(k << 28) | i for k, i in zip(enc["feat_kind"], enc["feat_idx"])], np.int32)

from decider2.trees.encode import EncodeContext, _infer_kinds, _walk, safe_ident
ctx = EncodeContext(tree, safe_ident(tree.name), kinds=_infer_kinds(tree, safe_ident(tree.name), ft))
start_pc = _walk(ctx)


@njit(inline="always")
def walk_separate(feats, thr_f, thr_i):
    f64, i64, b8, i32, s64, sbytes = feats
    pc = start_pc
    while True:
        k = kind[pc]
        if k == LEAF:
            return leaf_value[pc]
        fk = feat_kind[pc]
        j = feat_idx[pc]
        if k == CMP:
            o = op[pc]
            if fk == F64:
                r = compare(o, f64[j], thr_f[thr_slot[pc]])
            elif fk == I64:
                r = compare(o, i64[j], thr_i[thr_slot[pc]])
            elif fk == BOOL:
                r = compare(o, b8[j], thr_i[thr_slot[pc]] != 0)
            else:
                r = compare(o, i32[j], thr_i[thr_slot[pc]])
        else:
            if fk == F64:
                t = f64[j] != 0.0
            elif fk == I64:
                t = i64[j] != 0
            elif fk == BOOL:
                t = b8[j]
            else:
                t = i32[j] != 0
            r = t if k == IS_TRUE else not t
        pc = then_[pc] if r else else_[pc]


@njit(inline="always")
def walk_packed(feats, thr_f, thr_i):
    f64, i64, b8, i32, s64, sbytes = feats
    pc = start_pc
    while True:
        k = kind[pc]
        if k == LEAF:
            return leaf_value[pc]
        p = packed[pc]
        fk = p >> 28
        j = p & 0x0FFFFFFF
        if k == CMP:
            o = op[pc]
            if fk == F64:
                r = compare(o, f64[j], thr_f[thr_slot[pc]])
            elif fk == I64:
                r = compare(o, i64[j], thr_i[thr_slot[pc]])
            elif fk == BOOL:
                r = compare(o, b8[j], thr_i[thr_slot[pc]] != 0)
            else:
                r = compare(o, i32[j], thr_i[thr_slot[pc]])
        else:
            if fk == F64:
                t = f64[j] != 0.0
            elif fk == I64:
                t = i64[j] != 0
            elif fk == BOOL:
                t = b8[j]
            else:
                t = i32[j] != 0
            r = t if k == IS_TRUE else not t
        pc = then_[pc] if r else else_[pc]


def make_kernel(walker):
    @njit
    def kernel(arrays, thr_f, thr_i, n, out):
        f_cols, i_cols, b_cols, c_cols, s_cols, s_bytes = arrays
        bf = np.empty(n_f, np.float64)
        bi = np.empty(n_i, np.int64)
        bb = np.empty(n_b, np.bool_)
        bc = np.empty(n_c, np.int32)
        bs = np.empty(2 * n_s, np.int64)
        row = (bf, bi, bb, bc, bs, s_bytes)
        for i in range(n):
            _fill_array(f_cols, i, bf)
            _fill_array(i_cols, i, bi)
            _fill_array(b_cols, i, bb)
            _fill_array(c_cols, i, bc)
            out[i] = walker(row, thr_f, thr_i)
    return kernel


def timeit(k, reps=15):
    out = np.empty(n, np.int64)
    for _ in range(2):
        k(arrays, thr_f, thr_i, n, out)
    s = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        k(arrays, thr_f, thr_i, n, out)
        s.append((time.perf_counter_ns() - t0) / n)
    return statistics.median(s), min(s), out


ks, kp = make_kernel(walk_separate), make_kernel(walk_packed)
results = {}
# interleave, twice, so drift on a shared box hits both equally
for label, k in (("separate", ks), ("packed", kp), ("separate", ks), ("packed", kp)):
    med, best, out = timeit(k)
    results.setdefault(label, []).append((med, best, out))
    print(f"feat_kind {label:9s} median {med:6.1f} ns/row  min {best:6.1f}")
assert (results["separate"][0][2] == results["packed"][0][2]).all()
with (HERE / "measurements.jsonl").open("a") as f:
    for label, runs in results.items():
        f.write(json.dumps({
            "label": "after", "tree": variant, "rows": n,
            "metric": f"feat_kind_{label}_ns_per_row",
            "median": statistics.median(m for m, _, _ in runs), "min": min(b for _, b, _ in runs),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }) + "\n")
