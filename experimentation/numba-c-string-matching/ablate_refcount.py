"""Why did the typed walk cost ~365 ns/row flat? Ablation over the row-walk
body: which construct makes numba emit per-row NRT refcount traffic. Each
variant is explicit source (no generated code), cache=False, timed on the
same numeric depth-1 program over 1M rows. Also records the count of
NRT_incref/NRT_decref call sites in each variant's LLVM."""
import re, time
import numpy as np
import llvmlite.ir as ll
from numba import njit, types
from numba.extending import intrinsic
from call_ptr import call_match
from kernels import compare, LEAF, CMP, IS_TRUE, IS_FALSE, STR, STR_MASK, F64, I64, U8, I32, STRB, ERR_LEAF
import kernels as K
from results_io import record

@intrinsic
def load_i64(typingctx, addr_t):
    """*(int64*)addr -- raw load, no array object, no refcount."""
    if addr_t not in (types.uint64, types.int64): return None
    def codegen(context, builder, sig, args):
        p = builder.inttoptr(args[0], ll.IntType(64).as_pointer())
        return builder.load(p)
    return types.int64(addr_t), codegen

def make_variants():
    # V0: numeric only (no string node kinds at all)
    @njit(cache=False)
    def v0(f64s, i64s, u8s, i32s, thr_f64, thr_i64, kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value, out):
        for i in range(out.shape[0]):
            pc = 0
            while True:
                k = kind[pc]
                if k == LEAF:
                    out[i] = leaf_value[pc]; break
                elif k == CMP:
                    fk = feat_kind[pc]; fi = feat_idx[pc]
                    if fk == F64: cond = compare(op[pc], f64s[i, fi], thr_f64[thr_slot[pc]])
                    elif fk == I64: cond = compare(op[pc], i64s[i, fi], thr_i64[thr_slot[pc]])
                    elif fk == U8: cond = compare(op[pc], np.int64(u8s[i, fi]), thr_i64[thr_slot[pc]])
                    else: cond = compare(op[pc], np.int64(i32s[i, fi]), thr_i64[thr_slot[pc]])
                    pc = then_[pc] if cond else else_[pc]
                elif k == IS_TRUE: pc = then_[pc] if u8s[i, feat_idx[pc]] != 0 else else_[pc]
                else: pc = then_[pc] if u8s[i, feat_idx[pc]] == 0 else else_[pc]
    # V1: + STR_MASK
    @njit(cache=False)
    def v1(f64s, i64s, u8s, i32s, thr_f64, thr_i64, masks, kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value, out):
        for i in range(out.shape[0]):
            pc = 0
            while True:
                k = kind[pc]
                if k == LEAF:
                    out[i] = leaf_value[pc]; break
                elif k == CMP:
                    fk = feat_kind[pc]; fi = feat_idx[pc]
                    if fk == F64: cond = compare(op[pc], f64s[i, fi], thr_f64[thr_slot[pc]])
                    elif fk == I64: cond = compare(op[pc], i64s[i, fi], thr_i64[thr_slot[pc]])
                    elif fk == U8: cond = compare(op[pc], np.int64(u8s[i, fi]), thr_i64[thr_slot[pc]])
                    else: cond = compare(op[pc], np.int64(i32s[i, fi]), thr_i64[thr_slot[pc]])
                    pc = then_[pc] if cond else else_[pc]
                elif k == IS_TRUE: pc = then_[pc] if u8s[i, feat_idx[pc]] != 0 else else_[pc]
                elif k == IS_FALSE: pc = then_[pc] if u8s[i, feat_idx[pc]] == 0 else else_[pc]
                else:
                    code = i32s[i, feat_idx[pc]]; slot = thr_slot[pc]
                    if code < 0 or code >= masks.shape[1] or slot < 0 or slot >= masks.shape[0]:
                        out[i] = ERR_LEAF; break
                    pc = then_[pc] if masks[slot, code] != 0 else else_[pc]
    # V2: + STR via a TUPLE of arrays (what kernels.py does today)
    @njit(cache=False)
    def v2(f64s, i64s, u8s, i32s, str_offsets, str_values, thr_f64, thr_i64, handles, fn_table, kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value, out):
        for i in range(out.shape[0]):
            pc = 0
            while True:
                k = kind[pc]
                if k == LEAF:
                    out[i] = leaf_value[pc]; break
                elif k == CMP:
                    fk = feat_kind[pc]; fi = feat_idx[pc]
                    if fk == F64: cond = compare(op[pc], f64s[i, fi], thr_f64[thr_slot[pc]])
                    elif fk == I64: cond = compare(op[pc], i64s[i, fi], thr_i64[thr_slot[pc]])
                    elif fk == U8: cond = compare(op[pc], np.int64(u8s[i, fi]), thr_i64[thr_slot[pc]])
                    else: cond = compare(op[pc], np.int64(i32s[i, fi]), thr_i64[thr_slot[pc]])
                    pc = then_[pc] if cond else else_[pc]
                elif k == IS_TRUE: pc = then_[pc] if u8s[i, feat_idx[pc]] != 0 else else_[pc]
                elif k == IS_FALSE: pc = then_[pc] if u8s[i, feat_idx[pc]] == 0 else else_[pc]
                else:
                    fi = feat_idx[pc]; slot = thr_slot[pc]
                    if fi < 0 or fi >= len(str_offsets) or slot < 0 or slot >= handles.shape[0]:
                        out[i] = ERR_LEAF; break
                    offs = str_offsets[fi]; vals = str_values[fi]
                    if i + 1 >= offs.shape[0]:
                        out[i] = ERR_LEAF; break
                    start = offs[i]; end = offs[i + 1]
                    if start < 0 or end < start or end > vals.shape[0]:
                        out[i] = ERR_LEAF; break
                    rc = call_match(fn_table[0], handles[slot], np.uint64(vals.ctypes.data) + np.uint64(start), end - start)
                    if rc < 0:
                        out[i] = ERR_LEAF; break
                    pc = then_[pc] if rc == 1 else else_[pc]
    # V3: + STR via 2D padded arrays? no -- via a POINTER TABLE: str_ptrs[c] = (offsets addr, values addr, n_rows+1, n_bytes), raw loads
    @njit(cache=False)
    def v3(f64s, i64s, u8s, i32s, str_tab, thr_f64, thr_i64, handles, fn_table, kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value, out):
        for i in range(out.shape[0]):
            pc = 0
            while True:
                k = kind[pc]
                if k == LEAF:
                    out[i] = leaf_value[pc]; break
                elif k == CMP:
                    fk = feat_kind[pc]; fi = feat_idx[pc]
                    if fk == F64: cond = compare(op[pc], f64s[i, fi], thr_f64[thr_slot[pc]])
                    elif fk == I64: cond = compare(op[pc], i64s[i, fi], thr_i64[thr_slot[pc]])
                    elif fk == U8: cond = compare(op[pc], np.int64(u8s[i, fi]), thr_i64[thr_slot[pc]])
                    else: cond = compare(op[pc], np.int64(i32s[i, fi]), thr_i64[thr_slot[pc]])
                    pc = then_[pc] if cond else else_[pc]
                elif k == IS_TRUE: pc = then_[pc] if u8s[i, feat_idx[pc]] != 0 else else_[pc]
                elif k == IS_FALSE: pc = then_[pc] if u8s[i, feat_idx[pc]] == 0 else else_[pc]
                else:
                    fi = feat_idx[pc]; slot = thr_slot[pc]
                    if fi < 0 or fi >= str_tab.shape[0] or slot < 0 or slot >= handles.shape[0]:
                        out[i] = ERR_LEAF; break
                    off_addr = str_tab[fi, 0]; val_addr = str_tab[fi, 1]; n_off = str_tab[fi, 2]; n_bytes = str_tab[fi, 3]
                    if i + 1 >= n_off:
                        out[i] = ERR_LEAF; break
                    start = load_i64(off_addr + np.uint64(8 * i)); end = load_i64(off_addr + np.uint64(8 * (i + 1)))
                    if start < 0 or end < start or end > n_bytes:
                        out[i] = ERR_LEAF; break
                    rc = call_match(fn_table[0], handles[slot], val_addr + np.uint64(start), end - start)
                    if rc < 0:
                        out[i] = ERR_LEAF; break
                    pc = then_[pc] if rc == 1 else else_[pc]
    # Vf64: today's single-float64 walk in the SAME single-loop style (fair baseline)
    @njit(cache=False)
    def vf64(feats, thr, kind, feat_idx, op, thr_slot, then_, else_, leaf_value, out):
        for i in range(feats.shape[0]):
            pc = 0
            while True:
                k = kind[pc]
                if k == LEAF:
                    out[i] = leaf_value[pc]; break
                elif k == CMP:
                    pc = then_[pc] if compare(op[pc], feats[i, feat_idx[pc]], thr[thr_slot[pc]]) else else_[pc]
                elif k == IS_TRUE: pc = then_[pc] if feats[i, feat_idx[pc]] != 0.0 else else_[pc]
                else: pc = then_[pc] if feats[i, feat_idx[pc]] == 0.0 else else_[pc]
    # V4 / V5: V3's body as a separate per-row FUNCTION (decider2's walk_tree shape), real call vs IR-inlined
    def row_body(f64s, i64s, u8s, i32s, str_tab, i, thr_f64, thr_i64, handles, fn_table, kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value):
        pc = 0
        while True:
            k = kind[pc]
            if k == LEAF:
                return leaf_value[pc]
            elif k == CMP:
                fk = feat_kind[pc]; fi = feat_idx[pc]
                if fk == F64: cond = compare(op[pc], f64s[i, fi], thr_f64[thr_slot[pc]])
                elif fk == I64: cond = compare(op[pc], i64s[i, fi], thr_i64[thr_slot[pc]])
                elif fk == U8: cond = compare(op[pc], np.int64(u8s[i, fi]), thr_i64[thr_slot[pc]])
                else: cond = compare(op[pc], np.int64(i32s[i, fi]), thr_i64[thr_slot[pc]])
                pc = then_[pc] if cond else else_[pc]
            elif k == IS_TRUE: pc = then_[pc] if u8s[i, feat_idx[pc]] != 0 else else_[pc]
            elif k == IS_FALSE: pc = then_[pc] if u8s[i, feat_idx[pc]] == 0 else else_[pc]
            else:
                fi = feat_idx[pc]; slot = thr_slot[pc]
                if fi < 0 or fi >= str_tab.shape[0] or slot < 0 or slot >= handles.shape[0]:
                    return ERR_LEAF
                off_addr = str_tab[fi, 0]; val_addr = str_tab[fi, 1]; n_off = str_tab[fi, 2]; n_bytes = str_tab[fi, 3]
                if i + 1 >= n_off:
                    return ERR_LEAF
                start = load_i64(off_addr + np.uint64(8 * i)); end = load_i64(off_addr + np.uint64(8 * (i + 1)))
                if start < 0 or end < start or end > n_bytes:
                    return ERR_LEAF
                rc = call_match(fn_table[0], handles[slot], val_addr + np.uint64(start), end - start)
                if rc < 0:
                    return ERR_LEAF
                pc = then_[pc] if rc == 1 else else_[pc]
    row_call = njit(cache=False)(row_body)
    row_inl = njit(cache=False, inline="always")(row_body)
    @njit(cache=False)
    def v4(f64s, i64s, u8s, i32s, str_tab, thr_f64, thr_i64, handles, fn_table, kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value, out):
        for i in range(out.shape[0]):
            out[i] = row_call(f64s, i64s, u8s, i32s, str_tab, i, thr_f64, thr_i64, handles, fn_table, kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value)
    @njit(cache=False)
    def v5(f64s, i64s, u8s, i32s, str_tab, thr_f64, thr_i64, handles, fn_table, kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value, out):
        for i in range(out.shape[0]):
            out[i] = row_inl(f64s, i64s, u8s, i32s, str_tab, i, thr_f64, thr_i64, handles, fn_table, kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value)
    return v0, v1, v2, v3, vf64, v4, v5

def nrt_counts(disp):
    ir = list(disp.inspect_llvm().values())[0]
    return len(re.findall(r"call [^\n]*@NRT_incref", ir)), len(re.findall(r"call [^\n]*@NRT_decref", ir))

def best(fn, *a, reps=5):
    fn(*a); t=[]
    for _ in range(reps):
        t0=time.perf_counter(); fn(*a); t.append(time.perf_counter()-t0)
    return min(t)

def main():
    n = 1_000_000
    rng = np.random.default_rng(0)
    f64s = rng.uniform(0, 100, (n, 2)); i64s = np.zeros((n, 0), np.int64); u8s = np.zeros((n, 1), np.uint8); i32s = np.zeros((n, 1), np.int32)
    thr_f64 = np.array([50.0]); thr_i64 = np.zeros(0, np.int64)
    prog = K.Program([(CMP, F64, 0, 3, 0, 1, 2, 0), (LEAF, 0,0,0,0,0,0, 1), (LEAF, 0,0,0,0,0,0, 0)])
    out = np.zeros(n, np.int32)
    offs = np.zeros(n + 1, np.int64); vals = np.zeros(1, np.uint8)
    str_tab = np.array([[offs.ctypes.data, vals.ctypes.data, offs.shape[0], vals.shape[0]]], np.uint64)
    handles = np.zeros(1, np.int64); fn_table = np.zeros(1, np.uint64); masks = np.zeros((1, 1), np.uint8)
    v0, v1, v2, v3, vf64, v4, v5 = make_variants()
    progf = K.Program([(CMP, F64, 0, 3, 0, 1, 2, 0), (LEAF, 0,0,0,0,0,0, 1), (LEAF, 0,0,0,0,0,0, 0)])
    P = prog.arrays()
    runs = [
        ("Vf64 today's float64 walk, same single-loop style", vf64, (f64s, thr_f64, *progf.arrays_f64(), out)),
        ("V4 = V3 body as separate per-row njit function (real call)", v4, (f64s, i64s, u8s, i32s, str_tab, thr_f64, thr_i64, handles, fn_table, *P, out)),
        ("V5 = V3 body as per-row function, inline='always'", v5, (f64s, i64s, u8s, i32s, str_tab, thr_f64, thr_i64, handles, fn_table, *P, out)),
        ("V0 numeric only", v0, (f64s, i64s, u8s, i32s, thr_f64, thr_i64, *P, out)),
        ("V1 + STR_MASK node", v1, (f64s, i64s, u8s, i32s, thr_f64, thr_i64, masks, *P, out)),
        ("V2 + STR node, tuple-of-arrays (kernels.py today)", v2, (f64s, i64s, u8s, i32s, (offs,), (vals,), thr_f64, thr_i64, handles, fn_table, *P, out)),
        ("V3 + STR node, pointer table + raw loads", v3, (f64s, i64s, u8s, i32s, str_tab, thr_f64, thr_i64, handles, fn_table, *P, out)),
    ]
    for name, fn, args in runs:
        t = best(fn, *args)
        inc, dec = nrt_counts(fn)
        record("refcount_ablation", variant=name, ns_per_row=t / n * 1e9, nrt_incref_sites=inc, nrt_decref_sites=dec)

if __name__ == "__main__":
    main()
