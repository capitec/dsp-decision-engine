"""Where the per-call import cost goes at batch size 1: the incumbent's
ctypes loop (arrowc.export) piece by piece, and NanoView's Python-side
struct churn against the one C call it wraps. Also a pooled variant of the
nanoarrow import (structs allocated once, reused per call)."""
from __future__ import annotations

import ctypes
import statistics
import time

import numpy as np
import polars as pl

import nashim
import arrowc
from nashim import ERR_SIZE, VIEW_SIZE, NanoView, lib
from results_io import record
from bench import make_series


def tm(fn, k=2000, blocks=21):
    out = []
    j = 0
    for _ in range(blocks):
        t0 = time.perf_counter_ns()
        for _ in range(k):
            fn(j); j += 1
        out.append((time.perf_counter_ns() - t0) / k / 1000.0)
    return statistics.median(out), min(out)


def main():
    series = [make_series(1, seed) for seed in range(1000)]
    S = lambda j: series[j % 1000]
    res = {}

    def rec(stage, fn):
        med, mn = tm(fn)
        res[stage] = med
        record(probe="ablate_import", n=1, stage=stage, median_us=med, min_us=mn)
        print(f"{stage:<58} median {med:8.2f} us   min {mn:8.2f} us", flush=True)

    # ---- incumbent's export, step by step ----
    rec("stream: __arrow_c_stream__()", lambda j: S(j).__arrow_c_stream__())
    def cap_ptr(j):
        c = S(j).__arrow_c_stream__()
        return arrowc._PyCapsule_GetPointer(c, b"arrow_array_stream")
    rec("+ PyCapsule_GetPointer", cap_ptr)
    def ctypes_loop(j):
        c = S(j).__arrow_c_stream__()
        ptr = arrowc._PyCapsule_GetPointer(c, b"arrow_array_stream")
        stream = ctypes.cast(ptr, ctypes.POINTER(arrowc.ArrowArrayStream)).contents
        schema = arrowc.ArrowSchema()
        stream.get_schema(ctypes.byref(stream), ctypes.byref(schema))
        fmt = schema.format.decode()
        schema.release(ctypes.byref(schema))
        arr = arrowc.ArrowArray()
        stream.get_next(ctypes.byref(stream), ctypes.byref(arr))
        arr2 = arrowc.ArrowArray()
        stream.get_next(ctypes.byref(stream), ctypes.byref(arr2))
        nb = arr.n_buffers
        bufs = [arr.buffers[i] or 0 for i in range(nb)]
        arr.release(ctypes.byref(arr))
        stream.release(ctypes.byref(stream))
        return fmt, bufs
    rec("+ ctypes stream loop (schema, 2x get_next, buffers, release)", ctypes_loop)
    def with_sizes(j):
        fmt, bufs = ctypes_loop(j)
        n_data = len(bufs) - 3
        return list(np.ctypeslib.as_array(ctypes.cast(bufs[-1], ctypes.POINTER(ctypes.c_int64)), shape=(n_data,)))
    rec("+ np.ctypeslib.as_array over the sizes buffer", with_sizes)
    def with_sizes_ctypes(j):
        fmt, bufs = ctypes_loop(j)
        n_data = len(bufs) - 3
        return list((ctypes.c_int64 * n_data).from_address(bufs[-1]))
    rec("  (alt: sizes via (c_int64*n).from_address)", with_sizes_ctypes)
    rec("arrowc.export + release (as written)", lambda j: arrowc.export(S(j)).release())

    # ---- nanoarrow import, as written vs pooled ----
    rec("NanoView(s) + release (as written)", lambda j: NanoView(S(j)).release())
    schema = arrowc.ArrowSchema(); arr = arrowc.ArrowArray(); arr2 = arrowc.ArrowArray()
    view = ctypes.create_string_buffer(VIEW_SIZE); err = ctypes.create_string_buffer(ERR_SIZE)
    p_schema, p_arr, p_arr2 = ctypes.byref(schema), ctypes.byref(arr), ctypes.byref(arr2)
    def pooled(j):
        c = S(j).__arrow_c_stream__()
        ptr = arrowc._PyCapsule_GetPointer(c, b"arrow_array_stream")
        rc = lib.sm_import_single(ptr, p_schema, p_arr, p_arr2, view, err)
        assert rc == 0
        lib.sm_view_reset(view); lib.sm_array_release(p_arr); lib.sm_schema_release(p_schema)
    rec("pooled: stream + capsule + sm_import_single + 3 releases", pooled)
    def pooled_call_only(j):
        c = S(j).__arrow_c_stream__()
        ptr = arrowc._PyCapsule_GetPointer(c, b"arrow_array_stream")
        rc = lib.sm_import_single(ptr, p_schema, p_arr, p_arr2, view, err)
        # NOTE: leaks the array/schema (polars keeps a ref); measures the call itself
        arr.release = ctypes.cast(None, arrowc.ArrowArray._fields_[8][1]) if False else arr.release
        lib.sm_array_release(p_arr); lib.sm_schema_release(p_schema)
    rec("ctypes object churn: 3 structs + 2 buffers per call", lambda j: (
        arrowc.ArrowSchema(), arrowc.ArrowArray(), arrowc.ArrowArray(),
        ctypes.create_string_buffer(VIEW_SIZE), ctypes.create_string_buffer(ERR_SIZE)))


if __name__ == "__main__":
    main()
