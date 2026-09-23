"""Equal-effort floors at batch size 1: what A and B cost per call once the
obvious per-call waste is removed (pattern table hoisted for both; A's
sizes read via ctypes instead of np.ctypeslib.as_array; B's structs pooled
and reused). Neither changes the per-row code."""
from __future__ import annotations

import ctypes
import statistics
import sys
import time

import numpy as np

import nashim
import arrowc
from kernel import CONTAINS, LEAF, STR, build_tree, pattern_table, walk_chunk
from kernel_b import walk_chunk_b
from nashim import ERR_SIZE, GET_STRING_ADDR, VIEW_SIZE, lib
from results_io import record
from bench import make_series

TREE = build_tree([(STR, 0, CONTAINS, 0, 1, 2, 0), (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)])
PAT = pattern_table(["dog"])
T = TREE


def tm(fn, k=2000, blocks=25):
    out = []
    j = 0
    for _ in range(blocks):
        t0 = time.perf_counter_ns()
        for _ in range(k):
            fn(j); j += 1
        out.append((time.perf_counter_ns() - t0) / k / 1000.0)
    return statistics.median(out), min(out)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    series = [make_series(n, seed) for seed in range(1000)]
    S = lambda j: series[j % 1000]
    feats = np.empty((n, 0)); out = np.empty(n, np.int64)

    # ---- A lean: same kernel, same tables, cheaper Python around it ----
    P = ctypes.POINTER(arrowc.ArrowArrayStream)
    views = np.zeros(1, np.uint64); valid = np.zeros(1, np.uint64); valid_off = np.zeros(1, np.int64)
    nd = np.zeros(1, np.int64); base = np.zeros(1, np.int64)
    daddr = np.zeros(64, np.uint64); dsize = np.zeros(64, np.uint64)
    schema = arrowc.ArrowSchema(); arr = arrowc.ArrowArray(); arr2 = arrowc.ArrowArray()
    def a_lean(j):
        c = S(j).__arrow_c_stream__()
        stream = ctypes.cast(arrowc._PyCapsule_GetPointer(c, b"arrow_array_stream"), P).contents
        stream.get_schema(ctypes.byref(stream), ctypes.byref(schema))
        if schema.format != b"vu":
            raise ValueError(schema.format)
        schema.release(ctypes.byref(schema))
        stream.get_next(ctypes.byref(stream), ctypes.byref(arr))
        stream.get_next(ctypes.byref(stream), ctypes.byref(arr2))
        nb = arr.n_buffers; k = nb - 3
        bufs = arr.buffers
        views[0] = (bufs[1] or 0) + 16 * arr.offset; valid[0] = bufs[0] or 0; valid_off[0] = arr.offset
        nd[0] = k
        for q in range(k):
            daddr[q] = bufs[2 + q]
        dsize[:k] = (ctypes.c_int64 * k).from_address(bufs[nb - 1])
        walk_chunk(feats, views, valid, valid_off, nd, base, daddr, dsize, *PAT,
                   T["thresholds"], T["kind"], T["feat_idx"], T["op"], T["thr_slot"], T["then_"], T["else_"],
                   T["leaf_value"], out, 0, arr.length)
        arr.release(ctypes.byref(arr)); stream.release(ctypes.byref(stream))
    a_lean(0)
    med, mn = tm(a_lean)
    record(probe="bench_lean", n=n, stage="A.lean_end_to_end", median_us=med, min_us=mn)
    print(f"n={n} A.lean_end_to_end (ctypes loop, no numpy glue, hoisted pattern)   median {med:8.2f} us  min {mn:8.2f} us")

    # ---- B pooled: structs allocated once ----
    view = ctypes.create_string_buffer(VIEW_SIZE); err = ctypes.create_string_buffer(ERR_SIZE)
    p_schema, p_arr, p_arr2 = ctypes.byref(schema), ctypes.byref(arr), ctypes.byref(arr2)
    addrs = np.array([ctypes.addressof(view)], np.uint64); fn = np.uint64(GET_STRING_ADDR)
    def b_pooled(j):
        c = S(j).__arrow_c_stream__()
        ptr = arrowc._PyCapsule_GetPointer(c, b"arrow_array_stream")
        if lib.sm_import_single(ptr, p_schema, p_arr, p_arr2, view, err) != 0:
            raise RuntimeError(lib.sm_error_message(err))
        walk_chunk_b(feats, fn, addrs, *PAT, T["thresholds"], T["kind"], T["feat_idx"], T["op"], T["thr_slot"],
                     T["then_"], T["else_"], T["leaf_value"], out, 0, arr.length)
        lib.sm_view_reset(view); lib.sm_array_release(p_arr); lib.sm_schema_release(p_schema)
    b_pooled(0)
    med, mn = tm(b_pooled)
    record(probe="bench_lean", n=n, stage="B.pooled_end_to_end", median_us=med, min_us=mn)
    print(f"n={n} B.pooled_end_to_end (structs pooled, hoisted pattern)               median {med:8.2f} us  min {mn:8.2f} us")
    def c_pooled(j):
        c = S(j).__arrow_c_stream__()
        ptr = arrowc._PyCapsule_GetPointer(c, b"arrow_array_stream")
        if lib.sm_import_single(ptr, p_schema, p_arr, p_arr2, view, err) != 0:
            raise RuntimeError(lib.sm_error_message(err))
        if lib.sm_view_validate(view, 3, err) != 0:
            raise RuntimeError(lib.sm_error_message(err))
        if lib.sm_view_storage_type(view) != 41:  # NANOARROW_TYPE_STRING_VIEW
            raise ValueError("not utf8_view")
        nb = arr.n_buffers; k = nb - 3
        bufs = arr.buffers
        views[0] = (bufs[1] or 0) + 16 * arr.offset; valid[0] = bufs[0] or 0; valid_off[0] = arr.offset
        nd[0] = k
        for q in range(k):
            daddr[q] = bufs[2 + q]
        dsize[:k] = (ctypes.c_int64 * k).from_address(bufs[nb - 1])
        walk_chunk(feats, views, valid, valid_off, nd, base, daddr, dsize, *PAT,
                   T["thresholds"], T["kind"], T["feat_idx"], T["op"], T["thr_slot"], T["then_"], T["else_"],
                   T["leaf_value"], out, 0, arr.length)
        lib.sm_view_reset(view); lib.sm_array_release(p_arr); lib.sm_schema_release(p_schema)
    c_pooled(0)
    med, mn = tm(c_pooled)
    record(probe="bench_lean", n=n, stage="C-full.pooled_end_to_end", median_us=med, min_us=mn)
    print(f"n={n} C-full.pooled_end_to_end (pooled import + validate FULL + A kernel) median {med:8.2f} us  min {mn:8.2f} us")


if __name__ == "__main__":
    main()
