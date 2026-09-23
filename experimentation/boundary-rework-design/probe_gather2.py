"""Probe 2: the per-row cost of three gather shapes over the SAME imported
frame, 17 columns (10 f64, 4 i64, 2 bool, 1 string), 1M rows, plus n=1.
  V1  generic nanoarrow accessors per element (rs_gather_row)      -- probe 1
  V2  column descriptors resolved once in C, plain loads per row   (rs_gather_row2)
  V3  no per-row C call: numba loads through intrinsics from the
      addresses the shim resolved (validity bit read in numba)
  N   today's numpy columns + numba _fill (typed-features branch), string as code
Walk identical for all; answers asserted equal."""
from __future__ import annotations
import ctypes, statistics, time
import numpy as np
import polars as pl
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic
import rowshim as rs
from rowshim import F64, I64, BOOL, STR, call_gather, call_get_string, load_u8
from probe_gather import make_df, make_tree, walk, kernel_numpy, kernel_nano, PAT_ARR, NF, NI, NB, tm

_vp, _i64 = ctypes.c_void_p, ctypes.c_int64


class ColDesc(ctypes.Structure):
    _fields_ = [("validity", _vp), ("data", _vp), ("view", _vp), ("offset", _i64),
                ("kind", ctypes.c_int32), ("width", ctypes.c_int32), ("slot", ctypes.c_int32), ("is_signed", ctypes.c_int32)]


class RowPlan2(ctypes.Structure):
    _fields_ = [("ncols", ctypes.c_int32), ("cols", ctypes.POINTER(ColDesc)), ("f64", ctypes.POINTER(ctypes.c_double)),
                ("i64", ctypes.POINTER(_i64)), ("b8", ctypes.POINTER(ctypes.c_uint8)), ("spans", ctypes.POINTER(_i64)),
                ("valid", ctypes.POINTER(ctypes.c_uint8))]


lib = rs.lib
assert ctypes.sizeof(ColDesc) == lib.rs_sizeof_coldesc() and ctypes.sizeof(RowPlan2) == lib.rs_sizeof_plan2()
lib.rs_resolve_col.argtypes = [_vp, ctypes.c_int32, ctypes.c_int32, _vp]; lib.rs_resolve_col.restype = ctypes.c_int
lib.rs_gather_row2.argtypes = [_vp, _i64]; lib.rs_gather_row2.restype = None
lib.rs_plan2_addrs.argtypes = [_vp, _vp]; lib.rs_plan2_addrs.restype = None
lib.rs_resolve_all.argtypes = [_vp, _vp, _vp, _vp, ctypes.c_int32, _vp, _vp]; lib.rs_resolve_all.restype = ctypes.c_int
GATHER2_ADDR = ctypes.cast(lib.rs_gather_row2, _vp).value


@intrinsic
def load_f64(typingctx, addr):
    if not isinstance(addr, types.Integer): return None
    def codegen(context, builder, sig, args):
        return builder.load(builder.inttoptr(args[0], ir.DoubleType().as_pointer()))
    return types.float64(addr), codegen


@intrinsic
def load_i64(typingctx, addr):
    if not isinstance(addr, types.Integer): return None
    def codegen(context, builder, sig, args):
        return builder.load(builder.inttoptr(args[0], ir.IntType(64).as_pointer()))
    return types.int64(addr), codegen


@njit(inline="always")
def _bit(addr, j):
    return (load_u8(addr + np.uint64(j >> 3)) >> (j & 7)) & 1


# V3: numba loads straight from resolved addresses (kinds/slots are program data)
@njit
def kernel_v3(get_string_addr, addrs, kinds, slots, bf, bi, bb, spans, pat, thr_f, thr_i, tree, n, out):
    kind, fk, fi, op, thr, then_, else_, leaf_value = tree
    pat_addr = np.uint64(pat.ctypes.data); pat_len = len(pat)
    ncols = len(kinds)
    for i in range(n):
        for c in range(ncols):
            k = kinds[c]; s = slots[c]
            data = addrs[c, 0]; valid = addrs[c, 1]; off = addrs[c, 2]
            isnull = valid != 0 and _bit(valid, off + i) == 0
            if k == F64:
                bf[s] = np.nan if isnull else load_f64(data + np.uint64(8 * i))
            elif k == I64:
                bi[s] = 0 if isnull else load_i64(data + np.uint64(8 * i))
            elif k == BOOL:
                bb[s] = 0 if isnull else _bit(data, off + i)
            else:
                if isnull:
                    spans[2 * s] = 0; spans[2 * s + 1] = -1
                else:
                    a, ln = call_get_string(get_string_addr, addrs[c, 3], i)
                    spans[2 * s] = a; spans[2 * s + 1] = ln
        out[i] = walk(bf, bi, bb, spans, thr_f, thr_i, pat_addr, pat_len, kind, fk, fi, op, thr, then_, else_, leaf_value)


@njit
def kernel_v2(gather_addr, plan_addr, bf, bi, bb, spans, pat, thr_f, thr_i, tree, n, out):
    kind, fk, fi, op, thr, then_, else_, leaf_value = tree
    pat_addr = np.uint64(pat.ctypes.data); pat_len = len(pat)
    for i in range(n):
        call_gather(gather_addr, plan_addr, i)
        out[i] = walk(bf, bi, bb, spans, thr_f, thr_i, pat_addr, pat_len, kind, fk, fi, op, thr, then_, else_, leaf_value)


def main():
    tree, thr_f, thr_i = make_tree()
    names = [f"f{k}" for k in range(NF)] + [f"i{k}" for k in range(NI)] + [f"b{k}" for k in range(NB)] + ["s"]
    kinds = [F64] * NF + [I64] * NI + [BOOL] * NB + [STR]
    slots = list(range(NF)) + list(range(NI)) + list(range(NB)) + [0]
    nc = len(names)
    imp = rs.FrameImport()
    # pooled plans/buffers
    plan1, plan1_addr, (bf, bi, bb, bs, bv), keep1 = rs.make_plan([0] * nc, kinds, slots, NF, NI, NB, 1)
    cols = (ColDesc * nc)()
    plan2 = RowPlan2(nc, cols, bf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), bi.ctypes.data_as(ctypes.POINTER(_i64)),
                     bb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), bs.ctypes.data_as(ctypes.POINTER(_i64)),
                     bv.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    plan2_addr = ctypes.addressof(plan2)
    addrs = np.zeros((nc, 4), np.uint64)
    kinds_a = np.array(kinds, np.int32); slots_a = np.array(slots, np.int32)
    gs = np.uint64(rs.GET_STRING_ADDR)

    child_idx = np.arange(nc, dtype=np.int32)
    frame_view = imp._p[3]

    def resolve():
        rc = lib.rs_resolve_all(frame_view, kinds_a.ctypes.data, slots_a.ctypes.data, child_idx.ctypes.data, nc, plan2_addr, addrs.ctypes.data)
        assert rc == 0, rc

    def resolve_v1():
        for k in range(nc):
            plan1.views[k] = imp.child(k)

    def run_v1(df, out):
        imp.import_df(df); resolve_v1()
        kernel_nano(np.uint64(rs.GATHER_ADDR), np.uint64(plan1_addr), bf, bi, bb, bs, PAT_ARR, thr_f, thr_i, tree, len(out), out)
        imp.release()

    def run_v2(df, out):
        imp.import_df(df); resolve()
        kernel_v2(np.uint64(GATHER2_ADDR), np.uint64(plan2_addr), bf, bi, bb, bs, PAT_ARR, thr_f, thr_i, tree, len(out), out)
        imp.release()

    def run_v3(df, out):
        imp.import_df(df); resolve()
        kernel_v3(gs, addrs, kinds_a, slots_a, bf, bi, bb, bs, PAT_ARR, thr_f, thr_i, tree, len(out), out)
        imp.release()

    from probe_gather import main as _m  # noqa: F401  (reuse extract)
    import probe_gather as pg

    def extract_numpy(df):
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
        kernel_numpy(*extract_numpy(df), PAT_ARR, thr_f, thr_i, tree, len(out), out)

    print("shape                       n=1 (us)     n=1000 (us)    n=1M (us)   ns/row@1M")
    for label, fn in (("N  numpy+_fill (today)", run_numpy), ("V1 nanoarrow generic/elem", run_v1),
                      ("V2 C descriptors, C gather", run_v2), ("V3 C descriptors, numba loads", run_v3)):
        res = []
        for n in (1, 1000, 1_000_000):
            df = make_df(n)
            ref = np.empty(n, np.int64); run_numpy(df, ref)
            out = np.empty(n, np.int64); fn(df, out)
            assert (out == ref).all(), label
            reps = 2000 if n == 1 else (200 if n == 1000 else 3)
            blocks = 15 if n <= 1000 else 5
            res.append(tm(lambda: fn(df, out), reps, blocks)[0])
        print(f"{label:28s} {res[0]:10.2f} {res[1]:12.2f} {res[2]:14.0f} {res[2]*1000/1e6:10.1f}", flush=True)

    # kernel-only at 1M rows (boundary excluded): the honest per-row gather+walk cost
    n = 1_000_000; df = make_df(n); out = np.empty(n, np.int64)
    cols_np = extract_numpy(df)
    kn = tm(lambda: kernel_numpy(*cols_np, PAT_ARR, thr_f, thr_i, tree, n, out), 3, 5)[0]
    imp.import_df(df); resolve(); resolve_v1()
    k1 = tm(lambda: kernel_nano(np.uint64(rs.GATHER_ADDR), np.uint64(plan1_addr), bf, bi, bb, bs, PAT_ARR, thr_f, thr_i, tree, n, out), 3, 5)[0]
    k2 = tm(lambda: kernel_v2(np.uint64(GATHER2_ADDR), np.uint64(plan2_addr), bf, bi, bb, bs, PAT_ARR, thr_f, thr_i, tree, n, out), 3, 5)[0]
    k3 = tm(lambda: kernel_v3(gs, addrs, kinds_a, slots_a, bf, bi, bb, bs, PAT_ARR, thr_f, thr_i, tree, n, out), 3, 5)[0]
    imp.release()
    tb = tm(lambda: extract_numpy(df), 3, 5)[0]
    print(f"\nkernel only @1M (ns/row): numpy+_fill {kn*1000/n:.1f} | V1 {k1*1000/n:.1f} | V2 {k2*1000/n:.1f} | V3 {k3*1000/n:.1f};  today's boundary alone (17 cols incl. cast(Categorical)) {tb*1000/n:.1f} ns/row")

    # n=1 import breakdown for the frame (17 columns)
    df = make_df(1)
    def imp_only(): imp.import_df(df); imp.release()
    def imp_res(): imp.import_df(df); resolve(); imp.release()
    print(f"\nn=1: import 17-col frame + release {tm(imp_only, 5000)[0]:.2f} us;  + resolve 17 cols (ONE C call) {tm(imp_res, 5000)[0]:.2f} us")
    # polars side alone: how long does polars take to hand over the stream and the struct array?
    P = ctypes.POINTER
    import arrowc_min as am
    print(f"n=1: polars __arrow_c_stream__ + get_schema + get_next + releases only (no nanoarrow) {tm(lambda: am.walk_stream(df), 5000)[0]:.2f} us")


if __name__ == "__main__":
    main()
