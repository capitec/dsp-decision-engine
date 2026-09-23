"""Probe 1: whole-frame nanoarrow import + one C gather per row versus
today's per-column numpy extraction + numba `_fill_array` gather.

Tree: 16 numeric features (10 f64, 4 i64, 2 bool) + 1 string column with an
exact match at the root; a chain of CMP nodes so every feature is read.
Two shapes, identical answers asserted, timed at n = 1, 1000, 1_000_000.
"""
from __future__ import annotations
import ctypes, statistics, sys, time
import numpy as np
import polars as pl
from numba import njit
import rowshim as rs
from rowshim import F64, I64, BOOL, STR, call_gather, load_u8

LEAF, CMP, STRX = 0, 1, 2
LT, EQ = 0, 2

NF, NI, NB = 10, 4, 2


def make_df(n, seed=0):
    r = np.random.default_rng(seed)
    d = {}
    for k in range(NF): d[f"f{k}"] = r.random(n)
    for k in range(NI): d[f"i{k}"] = r.integers(0, 100, n)
    for k in range(NB): d[f"b{k}"] = r.random(n) < 0.5
    d["s"] = r.choice(["retail", "mining-and-longer-name", "agri", "services-x"], n)
    return pl.DataFrame(d)


def make_tree():
    # node 0: STR s == "retail" -> 1 else 2; nodes 1..16: CMP chain over every feature; leaves
    kind, fk, fi, op, thr, then_, else_ = [], [], [], [], [], [], []
    nodes = [(STRX, STR, 0, EQ, 0)]
    feats = [(F64, k) for k in range(NF)] + [(I64, k) for k in range(NI)] + [(BOOL, k) for k in range(NB)]
    for k, (t, j) in enumerate(feats):
        nodes.append((CMP, t, j, LT, k))
    nn = len(nodes)
    for k, (kd, t, j, o, s) in enumerate(nodes):
        kind.append(kd); fk.append(t); fi.append(j); op.append(o); thr.append(s)
        nxt = k + 1 if k + 1 < nn else nn  # last CMP -> leaf nn
        then_.append(nxt); else_.append(nn + 1)  # fail -> leaf nn+1
    # leaves
    for _ in range(2):
        kind.append(LEAF); fk.append(0); fi.append(0); op.append(0); thr.append(0); then_.append(0); else_.append(0)
    leaf_value = [0] * nn + [1, 2]
    arr = lambda x, dt=np.int32: np.array(x, dt)
    thr_f = tuple(0.99 for _ in range(NF)) + tuple(0.0 for _ in range(NI + NB))
    thr_i = tuple(0 for _ in range(NF)) + tuple(99 for _ in range(NI)) + tuple(2 for _ in range(NB))
    return (arr(kind), arr(fk), arr(fi), arr(op), arr(thr), arr(then_), arr(else_), arr(leaf_value, np.int64)), thr_f, thr_i


PAT = b"retail"
PAT_ARR = np.frombuffer(PAT, np.uint8).copy()


@njit(inline="always")
def _eq_at(s_addr, p_addr, n):
    for j in range(n):
        if load_u8(s_addr + np.uint64(j)) != load_u8(p_addr + np.uint64(j)):
            return False
    return True


@njit(inline="always")
def walk(f64, i64, b8, spans, thr_f, thr_i, pat_addr, pat_len, kind, fk, fi, op, thr, then_, else_, leaf_value):
    pc = 0
    while True:
        k = kind[pc]
        if k == LEAF:
            return leaf_value[pc]
        j = fi[pc]
        if k == CMP:
            t = fk[pc]
            if t == F64:
                r = f64[j] < thr_f[thr[pc]]
            elif t == I64:
                r = i64[j] < thr_i[thr[pc]]
            else:
                r = b8[j] < thr_i[thr[pc]]
        else:  # STRX exact
            ln = spans[2 * j + 1]
            r = ln == pat_len and _eq_at(np.uint64(spans[2 * j]), pat_addr, pat_len)
        pc = then_[pc] if r else else_[pc]


# ---- shape A: today's typed gather from numpy columns (typed-features branch) ----
@njit(cache=True)
def _fill(arrays, i, out):
    for k in range(len(out)):
        out[k] = arrays[k][i]


@njit
def kernel_numpy(f_cols, i_cols, b_cols, codes, code_of_pat, pat, thr_f, thr_i, tree, n, out):
    kind, fk, fi, op, thr, then_, else_, leaf_value = tree
    bf = np.empty(NF, np.float64); bi = np.empty(NI, np.int64); bb = np.empty(NB, np.uint8)
    spans = np.empty(2, np.int64)
    pat_addr = np.uint64(pat.ctypes.data); pat_len = len(pat)
    for i in range(n):
        _fill(f_cols, i, bf); _fill(i_cols, i, bi); _fill(b_cols, i, bb)
        # today's string: a dictionary code compared to the pattern's code; the
        # span test is then a compare of the pattern with itself (same cost class)
        spans[1] = pat_len if codes[i] == code_of_pat else -1
        spans[0] = np.int64(pat_addr)
        out[i] = walk(bf, bi, bb, spans, thr_f, thr_i, pat_addr, pat_len, kind, fk, fi, op, thr, then_, else_, leaf_value)


# ---- shape B: one C gather per row through nanoarrow ----
@njit
def kernel_nano(gather_addr, plan_addr, bf, bi, bb, spans, pat, thr_f, thr_i, tree, n, out):
    kind, fk, fi, op, thr, then_, else_, leaf_value = tree
    pat_addr = np.uint64(pat.ctypes.data); pat_len = len(pat)
    for i in range(n):
        call_gather(gather_addr, plan_addr, i)
        out[i] = walk(bf, bi, bb, spans, thr_f, thr_i, pat_addr, pat_len, kind, fk, fi, op, thr, then_, else_, leaf_value)


def tm(fn, reps, blocks=15):
    outs = []
    for _ in range(blocks):
        t0 = time.perf_counter_ns()
        for _ in range(reps):
            fn()
        outs.append((time.perf_counter_ns() - t0) / reps / 1000.0)
    return statistics.median(outs), min(outs)


def main():
    tree, thr_f, thr_i = make_tree()
    names = [f"f{k}" for k in range(NF)] + [f"i{k}" for k in range(NI)] + [f"b{k}" for k in range(NB)] + ["s"]
    kinds = [F64] * NF + [I64] * NI + [BOOL] * NB + [STR]
    slots = list(range(NF)) + list(range(NI)) + list(range(NB)) + [0]
    imp = rs.FrameImport()

    def run_nano(df, out):
        imp.import_df(df)
        addrs = [imp.child(k) for k in range(len(names))]
        plan, plan_addr, (bf, bi, bb, bs, bv), keep = make_plan(addrs, kinds, slots)
        kernel_nano(np.uint64(rs.GATHER_ADDR), np.uint64(plan_addr), bf, bi, bb, bs, PAT_ARR, thr_f, thr_i, tree, len(out), out)
        imp.release()

    # pooled variant: plan/buffers built once, only child addresses refreshed
    plan0, plan_addr0, bufs0, keep0 = make_plan([0] * len(names), kinds, slots)
    bf0, bi0, bb0, bs0, bv0 = bufs0

    def run_nano_pooled(df, out):
        imp.import_df(df)
        for k in range(len(names)):
            plan0.views[k] = imp.child(k)
        kernel_nano(np.uint64(rs.GATHER_ADDR), np.uint64(plan_addr0), bf0, bi0, bb0, bs0, PAT_ARR, thr_f, thr_i, tree, len(out), out)
        imp.release()

    def extract_numpy(df):
        # today's boundary, approximated: values buffer -> numpy (zero-copy), string -> categorical codes
        f_cols = tuple(df[f"f{k}"]._get_buffers()["values"].to_numpy(allow_copy=False) for k in range(NF))
        i_cols = tuple(df[f"i{k}"]._get_buffers()["values"].to_numpy(allow_copy=False) for k in range(NI))
        b_cols = tuple(df[f"b{k}"]._get_buffers()["values"].to_numpy() for k in range(NB))
        enc = df["s"].cast(pl.Categorical)
        codes = enc._get_buffers()["values"].to_numpy(allow_copy=False)
        cats = enc.cat.get_categories().to_list()
        code_of_pat = cats.index("retail") if "retail" in cats else -1
        for c in f_cols + i_cols + b_cols: c.flags.writeable = False
        return f_cols, i_cols, b_cols, codes, code_of_pat

    def run_numpy(df, out):
        f_cols, i_cols, b_cols, codes, code_of_pat = extract_numpy(df)
        kernel_numpy(f_cols, i_cols, b_cols, codes, code_of_pat, PAT_ARR, thr_f, thr_i, tree, len(out), out)

    for n in (1, 1000, 1_000_000):
        df = make_df(n)
        outA = np.empty(n, np.int64); outB = np.empty(n, np.int64); outC = np.empty(n, np.int64)
        run_numpy(df, outA); run_nano(df, outB); run_nano_pooled(df, outC)
        assert (outA == outB).all() and (outA == outC).all(), (outA[:10], outB[:10])
        reps = 2000 if n == 1 else (200 if n == 1000 else 3)
        blocks = 15 if n <= 1000 else 5
        a = tm(lambda: run_numpy(df, outA), reps, blocks)
        b = tm(lambda: run_nano(df, outB), reps, blocks)
        c = tm(lambda: run_nano_pooled(df, outC), reps, blocks)
        print(f"n={n:>8}  today numpy+fill  {a[0]:10.2f} us  | nano frame+gather {b[0]:10.2f} us | nano pooled {c[0]:10.2f} us   (min {a[1]:.2f} / {b[1]:.2f} / {c[1]:.2f})", flush=True)
        if n == 1_000_000:
            print(f"           per row: numpy {a[0]*1000/n:.1f} ns  nano {c[0]*1000/n:.1f} ns")

    # breakdown at n=1
    df = make_df(1); out = np.empty(1, np.int64)
    t_stream = tm(lambda: df.__arrow_c_stream__(), 5000)
    def imp_only():
        imp.import_df(df); imp.release()
    t_imp = tm(imp_only, 5000)
    def imp_child():
        imp.import_df(df)
        for k in range(len(names)): plan0.views[k] = imp.child(k)
        imp.release()
    t_impc = tm(imp_child, 5000)
    imp.import_df(df)
    for k in range(len(names)): plan0.views[k] = imp.child(k)
    t_kernel = tm(lambda: kernel_nano(np.uint64(rs.GATHER_ADDR), np.uint64(plan_addr0), bf0, bi0, bb0, bs0, PAT_ARR, thr_f, thr_i, tree, 1, out), 5000)
    imp.release()
    t_extract = tm(lambda: extract_numpy(df), 2000)
    cols = extract_numpy(df)
    t_kernelA = tm(lambda: kernel_numpy(*cols, PAT_ARR, thr_f, thr_i, tree, 1, out), 5000)
    t_df = tm(lambda: make_df(1), 500)
    rec = {c: df[c][0] for c in df.columns}
    t_df_rec = tm(lambda: pl.DataFrame([rec]), 2000)
    print("\nn=1 breakdown (us, median):")
    print(f"  df.__arrow_c_stream__()                  {t_stream[0]:7.2f}")
    print(f"  + rs_import_frame + release (pooled)     {t_imp[0]:7.2f}")
    print(f"  + 17x rs_child (ctypes)                  {t_impc[0]:7.2f}")
    print(f"  kernel_nano dispatch + 1 row (12 args)   {t_kernel[0]:7.2f}")
    print(f"  today: 17-column numpy extraction        {t_extract[0]:7.2f}")
    print(f"  today: kernel_numpy dispatch + 1 row     {t_kernelA[0]:7.2f}")
    print(f"  pl.DataFrame([record]) (17 cols)         {t_df_rec[0]:7.2f}")


def make_plan(addrs, kinds, slots):
    return rs.make_plan(addrs, kinds, slots, NF, NI, NB, 1)


if __name__ == "__main__":
    main()
