"""The two walkers this experiment compares, plus the program encoding.

`walk_f64` is TODAY's shape (decider2/src/decider2/trees/interpreter.py
`walk_tree`, verbatim in spirit): every feature is one float64 slot, a
string test has already been hoisted to a bool feature at the frame tier
(cast to float64), and the node kinds are LEAF / CMP / IS_TRUE / IS_FALSE.

`walk_typed` is the proposal: features arrive as SEPARATE typed arrays
(float64, int64, uint8/bool, int32 codes, and a pointer table over Arrow
`offsets`+`values` for raw strings); every node carries a `feat_kind` beside its `feat_idx`, so a
node says "read slot 7 of the int64 array" or "match slot 3 of the string
buffers". Two new node kinds: STR (call C at the node, lazily) and
STR_MASK (per-category lookup: mask[pattern][code]).

Both are `@njit(cache=True)`. The C matcher's address (uint64) and the compiled
pattern ids (int64) are ARGUMENTS (`fn_table`, `handles`), never
globals -- §W's rule, and cache_check.py proves it holds.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from call_ptr import call_match, load_i64

# comparison opcodes -- decider2's six, same numbering as interpreter.py
LT, LE, EQ, GT, GE, NE = 0, 1, 2, 3, 4, 5
# node kinds -- decider2's four, plus two for strings
LEAF, CMP, IS_TRUE, IS_FALSE, STR, STR_MASK = 0, 1, 2, 3, 4, 5
# feature kinds -- which typed array a node reads
F64, I64, U8, I32, STRB = 0, 1, 2, 3, 4
# row result when a guard trips inside the kernel (never a crash)
ERR_LEAF = -1


@njit(cache=True, inline="always")
def compare(op, a, b):
    if op == LT:
        return a < b
    elif op == LE:
        return a <= b
    elif op == EQ:
        return a == b
    elif op == GT:
        return a > b
    elif op == GE:
        return a >= b
    else:
        return a != b


# ---------------------------------------------------------------------------
# Today: one float64 array. `feats` is row-major [n_rows, n_feats].
#
# SHAPE NOTE (measured, ablate_refcount.py): both walkers are written as ONE
# function with the row loop inside and `break` at the leaf. Written as a
# separate per-row function taking these arrays (called, or inline="always"),
# numba treats every array parameter as a fresh local and emits NRT
# incref/decref per row -- a flat ~365 ns/row that swamped everything
# (results_run1_polluted.jsonl). decider2's own `walk_tree` avoids it because
# its per-row `feats` is a tuple of SCALARS; a typed walker must keep string
# buffers as raw integers (`str_tab`) for the same reason.
# ---------------------------------------------------------------------------
@njit(cache=True)
def walk_f64(feats, thr, kind, feat_idx, op, thr_slot, then_, else_, leaf_value, out):
    for i in range(feats.shape[0]):
        pc = 0
        while True:
            k = kind[pc]
            if k == LEAF:
                out[i] = leaf_value[pc]
                break
            elif k == CMP:
                pc = then_[pc] if compare(op[pc], feats[i, feat_idx[pc]], thr[thr_slot[pc]]) else else_[pc]
            elif k == IS_TRUE:
                pc = then_[pc] if feats[i, feat_idx[pc]] != 0.0 else else_[pc]
            else:
                pc = then_[pc] if feats[i, feat_idx[pc]] == 0.0 else else_[pc]


# ---------------------------------------------------------------------------
# Proposal: typed arrays + feat_kind discriminator + lazy string node.
#   f64s [n, nf]  i64s [n, ni]  u8s [n, nb]  i32s [n, nc]
#   str_tab [n_string_cols, 4] uint64: (offsets addr, values addr, n_offsets,
#       n_bytes) per string column -- polars' Arrow buffers, zero-copy, read
#       through a raw-load intrinsic; the caller keeps the Series alive.
#   handles [n_patterns] int64 pattern IDS (C owns the pointers)
#   fn_table [1] uint64 (= address of sm_match_id)
#   masks [n_patterns, max_categories] uint8 for STR_MASK nodes.
# ---------------------------------------------------------------------------
@njit(cache=True)
def walk_typed(f64s, i64s, u8s, i32s, str_tab,
               thr_f64, thr_i64, handles, fn_table, masks,
               kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value, out):
    for i in range(out.shape[0]):
        pc = 0
        while True:
            k = kind[pc]
            if k == LEAF:
                out[i] = leaf_value[pc]
                break
            elif k == CMP:
                fk = feat_kind[pc]
                fi = feat_idx[pc]
                if fk == F64:
                    cond = compare(op[pc], f64s[i, fi], thr_f64[thr_slot[pc]])
                elif fk == I64:
                    cond = compare(op[pc], i64s[i, fi], thr_i64[thr_slot[pc]])
                elif fk == U8:
                    cond = compare(op[pc], np.int64(u8s[i, fi]), thr_i64[thr_slot[pc]])
                else:  # I32 codes
                    cond = compare(op[pc], np.int64(i32s[i, fi]), thr_i64[thr_slot[pc]])
                pc = then_[pc] if cond else else_[pc]
            elif k == IS_TRUE:
                pc = then_[pc] if u8s[i, feat_idx[pc]] != 0 else else_[pc]
            elif k == IS_FALSE:
                pc = then_[pc] if u8s[i, feat_idx[pc]] == 0 else else_[pc]
            elif k == STR:
                # The lazy node. These guards are the ONLY per-row checks in
                # the walk; they sit here because this is the one place an
                # index turns into a raw pointer handed to C.
                fi = feat_idx[pc]
                slot = thr_slot[pc]
                if fi < 0 or fi >= str_tab.shape[0] or slot < 0 or slot >= handles.shape[0]:
                    out[i] = ERR_LEAF
                    break
                off_addr = str_tab[fi, 0]
                val_addr = str_tab[fi, 1]
                if np.uint64(i + 1) >= str_tab[fi, 2]:
                    out[i] = ERR_LEAF
                    break
                start = load_i64(off_addr + np.uint64(8 * i))
                end = load_i64(off_addr + np.uint64(8 * (i + 1)))
                if start < 0 or end < start or np.uint64(end) > str_tab[fi, 3]:
                    out[i] = ERR_LEAF
                    break
                rc = call_match(fn_table[0], handles[slot], val_addr + np.uint64(start), end - start)
                if rc < 0:
                    out[i] = ERR_LEAF
                    break
                pc = then_[pc] if rc == 1 else else_[pc]
            else:  # STR_MASK: per-category -- code -> precomputed mask row
                code = i32s[i, feat_idx[pc]]
                slot = thr_slot[pc]
                if code < 0 or code >= masks.shape[1] or slot < 0 or slot >= masks.shape[0]:
                    out[i] = ERR_LEAF
                    break
                pc = then_[pc] if masks[slot, code] != 0 else else_[pc]


# ---------------------------------------------------------------------------
# Program encoding. A node is (kind, feat_kind, feat_idx, op, thr_slot, then, else, leaf).
# ---------------------------------------------------------------------------
class Program:
    def __init__(self, nodes):
        cols = list(zip(*nodes))
        self.kind, self.feat_kind, self.feat_idx, self.op, self.thr_slot, self.then_, self.else_, self.leaf_value = [
            np.asarray(c, dtype=np.int32) for c in cols
        ]

    def arrays(self):
        return (self.kind, self.feat_kind, self.feat_idx, self.op, self.thr_slot,
                self.then_, self.else_, self.leaf_value)

    def arrays_f64(self):
        """Today's walker has no feat_kind column."""
        return (self.kind, self.feat_idx, self.op, self.thr_slot, self.then_, self.else_, self.leaf_value)

    def validate(self, n_f64, n_i64, n_u8, n_i32, n_str, n_thr_f64, n_thr_i64, n_handles, n_masks):
        """Build-time validation -- where an out-of-range index SHOULD die,
        before any kernel runs. Raises ValueError with the node index."""
        n = len(self.kind)
        limits = {F64: n_f64, I64: n_i64, U8: n_u8, I32: n_i32, STRB: n_str}
        for pc in range(n):
            k = self.kind[pc]
            if k == LEAF:
                continue
            if k in (CMP, IS_TRUE, IS_FALSE, STR, STR_MASK):
                for tgt in (self.then_[pc], self.else_[pc]):
                    if not 0 <= tgt < n:
                        raise ValueError(f"node {pc}: jump target {tgt} outside program of {n} nodes")
            fk = self.feat_kind[pc]
            if fk not in limits:
                raise ValueError(f"node {pc}: unknown feat_kind {fk}")
            if not 0 <= self.feat_idx[pc] < limits[fk]:
                raise ValueError(f"node {pc}: feat_idx {self.feat_idx[pc]} out of range for kind {fk} (have {limits[fk]})")
            if k == CMP:
                lim = n_thr_f64 if fk == F64 else n_thr_i64
                if not 0 <= self.thr_slot[pc] < lim:
                    raise ValueError(f"node {pc}: threshold slot {self.thr_slot[pc]} out of range ({lim})")
            if k == STR and not 0 <= self.thr_slot[pc] < n_handles:
                raise ValueError(f"node {pc}: pattern slot {self.thr_slot[pc]} out of range ({n_handles} handles)")
            if k == STR_MASK and not 0 <= self.thr_slot[pc] < n_masks:
                raise ValueError(f"node {pc}: mask slot {self.thr_slot[pc]} out of range ({n_masks} masks)")
        return self


def owner_shape(third_node):
    """`ft1 > c1 AND ft2 > c2 AND <third>`, AND short-circuiting by
    construction: pc0 fails -> LEAF 0 without ever touching pc1/pc2.
    `third_node` is the pc2 tuple: STR / STR_MASK / IS_TRUE / CMP."""
    return Program([
        (CMP, F64, 0, GT, 0, 1, 3, 0),   # pc0: ft1 > thr_f64[0]
        (CMP, F64, 1, GT, 1, 2, 3, 0),   # pc1: ft2 > thr_f64[1]
        third_node,                      # pc2: -> pc4 if true else pc3
        (LEAF, 0, 0, 0, 0, 0, 0, 0),     # pc3: result 0
        (LEAF, 0, 0, 0, 0, 0, 0, 1),     # pc4: result 1
    ])


def empty_typed_inputs(n_rows):
    """Placeholders for typed arrays a program does not use -- numba needs
    a concrete type for every argument; zero-width arrays are free."""
    return dict(
        f64s=np.zeros((n_rows, 0)), i64s=np.zeros((n_rows, 0), np.int64),
        u8s=np.zeros((n_rows, 0), np.uint8), i32s=np.zeros((n_rows, 0), np.int32),
        str_tab=np.zeros((0, 4), np.uint64),
        thr_f64=np.zeros(0), thr_i64=np.zeros(0, np.int64),
        handles=np.zeros(0, np.int64), fn_table=np.zeros(1, np.uint64),
        masks=np.zeros((0, 0), np.uint8),
    )


def string_table(columns):
    """[(offsets, values), ...] -> the uint64 [n_cols, 4] pointer table the
    kernel reads. The caller MUST keep the underlying arrays/Series alive
    for as long as the table is in use (it holds raw addresses)."""
    tab = np.zeros((len(columns), 4), np.uint64)
    for c, (offs, vals) in enumerate(columns):
        assert offs.dtype == np.int64 and vals.dtype == np.uint8 and offs.flags.c_contiguous and vals.flags.c_contiguous
        tab[c] = (offs.ctypes.data, vals.ctypes.data, offs.shape[0], vals.shape[0])
    return tab


def string_buffers(series):
    """polars string column -> Arrow (offsets int64[n+1], values uint8[]) --
    the measured fact this experiment builds on. Returned as numpy views."""
    b = series._get_buffers()
    return b["offsets"].to_numpy(), b["values"].to_numpy()
