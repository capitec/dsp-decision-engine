"""Cost of nanoarrow's ArrowArrayViewValidate at each level as a function of
array length, and of the import handshake itself. If validation is O(rows)
it is paid in full on every single-record call under approach C."""
from __future__ import annotations

import statistics
import time

import numpy as np
import polars as pl

import nashim  # noqa: F401
from nashim import DEFAULT, FULL, MINIMAL, NanoView, lib
from results_io import record
from bench import make_series


def tm(fn, k, blocks):
    out = []
    for _ in range(blocks):
        t0 = time.perf_counter_ns()
        for _ in range(k):
            fn()
        out.append((time.perf_counter_ns() - t0) / k / 1000.0)
    return statistics.median(out), min(out)


def main():
    for n in (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000):
        s = make_series(n, 1)
        nv = NanoView(s)
        k = 2000 if n <= 1000 else (200 if n <= 100_000 else 5)
        for lvl, name in ((MINIMAL, "minimal"), (DEFAULT, "default"), (FULL, "full")):
            med, mn = tm(lambda: nv.validate(lvl), k, 15)
            record(probe="validate_cost", n=n, level=name, median_us=med, min_us=mn, ns_per_row=med * 1000 / n)
            print(f"n={n:>8} validate {name:<8} median {med:9.3f} us  ({med*1000/n:8.3f} ns/row)", flush=True)
        # the ctypes call floor: a no-op-ish C call through ctypes
        med, mn = tm(lambda: lib.sm_view_length(nv.chunks[0].view), k, 15)
        record(probe="validate_cost", n=n, level="ctypes_call_floor", median_us=med, min_us=mn)
        print(f"n={n:>8} ctypes call floor   median {med:9.3f} us", flush=True)
        nv.release()
        # also: a Utf8 ("u") column of the same length, if we can make one -- see spec_change.py


if __name__ == "__main__":
    main()
