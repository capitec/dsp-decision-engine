"""Typed-array walk vs today's single-float64 walk on a NUMERIC-ONLY tree
(no strings at all), so the only difference is the `feat_kind` switch --
the extra indirection the owner asked to have priced. Plus the int64
precision case: scaled-int64 money above 2**53 now compares correctly.
"""
import os, time
import numpy as np
BOUNDSCHECK = os.environ.get("NUMBA_BOUNDSCHECK", "0") == "1"
import kernels as K
from kernels import *
from results_io import record

N = 1_000_000
REPEATS = 7
TYPED_KEYS = ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")

def best(fn, repeats=REPEATS):
    fn(); t = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)

def random_tree(depth, n_f64, n_i64, n_u8, rng):
    """Full binary tree of CMP nodes over mixed-type features; leaves are
    result ids. Returns (typed nodes, f64-coerced nodes, thr_f64, thr_i64)
    encoding the SAME tree two ways."""
    typed, coerced = [], []
    thr_f64, thr_i64, thr_all = [], [], []
    feats = [(F64, i) for i in range(n_f64)] + [(I64, i) for i in range(n_i64)] + [(U8, i) for i in range(n_u8)]
    def build(d):
        pc = len(typed)
        if d == depth:
            typed.append((LEAF, 0, 0, 0, 0, 0, 0, pc)); coerced.append((LEAF, 0, 0, 0, 0, 0, 0, pc)); return pc
        fk, fi = feats[rng.integers(len(feats))]
        op = int(rng.integers(0, 6))
        typed.append(None); coerced.append(None)
        if fk == F64:
            v = float(rng.uniform(0, 100)); slot = len(thr_f64); thr_f64.append(v)
        elif fk == I64:
            v = int(rng.integers(0, 100_000)); slot = len(thr_i64); thr_i64.append(v)
        else:
            v = int(rng.integers(0, 2)); slot = len(thr_i64); thr_i64.append(v)
        slot_all = len(thr_all); thr_all.append(float(v))
        # coerced layout: f64 cols [0,n_f64), i64 cols next, u8 last
        col = fi if fk == F64 else (n_f64 + fi if fk == I64 else n_f64 + n_i64 + fi)
        t = build(d + 1); e = build(d + 1)
        typed[pc] = (CMP, fk, fi, op, slot, t, e, 0)
        coerced[pc] = (CMP, F64, col, op, slot_all, t, e, 0)
        return pc
    build(0)
    return K.Program(typed), K.Program(coerced), np.array(thr_f64), np.array(thr_i64, np.int64), np.array(thr_all)

def main():
    rng = np.random.default_rng(3)
    n_f64, n_i64, n_u8 = 4, 2, 2
    f64s = rng.uniform(0, 100, (N, n_f64))
    i64s = rng.integers(0, 100_000, (N, n_i64)).astype(np.int64)
    u8s = rng.integers(0, 2, (N, n_u8)).astype(np.uint8)
    coerced = np.column_stack([f64s, i64s.astype(np.float64), u8s.astype(np.float64)])
    out_t = np.zeros(N, np.int32); out_f = np.zeros(N, np.int32)
    for depth in (1, 3, 5, 8):
        progT, progF, thr_f64, thr_i64, thr_all = random_tree(depth, n_f64, n_i64, n_u8, rng)
        progT.validate(n_f64, n_i64, n_u8, 0, 0, len(thr_f64), len(thr_i64), 0, 0)
        inp = K.empty_typed_inputs(N); inp.update(f64s=f64s, i64s=i64s, u8s=u8s, thr_f64=thr_f64, thr_i64=thr_i64)
        t_typed = best(lambda: K.walk_typed(*[inp[k] for k in TYPED_KEYS], *progT.arrays(), out_t))
        t_f64 = best(lambda: K.walk_f64(coerced, thr_all, *progF.arrays_f64(), out_f))
        assert (out_t == out_f).all()
        record("typed_vs_f64_walk", boundscheck=BOUNDSCHECK, depth=depth, n_nodes=len(progT.kind), n=N,
               f64_walk_ns_per_row=t_f64 / N * 1e9, typed_walk_ns_per_row=t_typed / N * 1e9,
               typed_over_f64=t_typed / t_f64, identical=True)

    # ---- int64 precision: scaled-int64 money above 2**53 (doc 03 §1.2) ----
    big = np.array([[2**53 + 1], [2**53], [2**53 - 1], [9_007_199_254_740_993]], np.int64)  # cents
    thr = 2**53
    # typed: CMP I64 EQ/GT against an int64 threshold
    inp = K.empty_typed_inputs(4); inp.update(i64s=big, thr_i64=np.array([thr], np.int64))
    res = {}
    for opname, op in (("EQ", EQ), ("GT", GT)):
        prog = K.Program([(CMP, I64, 0, op, 0, 1, 2, 0), (LEAF, 0,0,0,0,0,0, 1), (LEAF, 0,0,0,0,0,0, 0)])
        o = np.zeros(4, np.int32); K.walk_typed(*[inp[k] for k in TYPED_KEYS], *prog.arrays(), o)
        progf = K.Program([(CMP, F64, 0, op, 0, 1, 2, 0), (LEAF, 0,0,0,0,0,0, 1), (LEAF, 0,0,0,0,0,0, 0)])
        of = np.zeros(4, np.int32); K.walk_f64(big.astype(np.float64), np.array([float(thr)]), *progf.arrays_f64(), of)
        truth = np.array([(v == thr) if op == EQ else (v > thr) for v in big[:, 0].tolist()], np.int32)
        res[opname] = dict(values=big[:, 0].tolist(), threshold=thr, truth=truth.tolist(),
                           f64_walk=of.tolist(), typed_walk=o.tolist(),
                           f64_correct=bool((of == truth).all()), typed_correct=bool((o == truth).all()))
        record("int64_precision", op=opname, **res[opname])

if __name__ == "__main__":
    main()
