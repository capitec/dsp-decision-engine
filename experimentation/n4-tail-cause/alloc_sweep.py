"""
Direct test of hypothesis 1 (allocator behaviour / mmap threshold) from the
N4 tail-cause brief: "vary allocation per call and watch the tail."

tail_cause.py's rusage instrumentation showed ru_minflt == ru_majflt == 0 for
EVERY one of 40,000 steady-state score() calls (both pinned and unpinned) --
i.e. literally no page faults at score()'s ~49KB/call allocation shape after
warmup. Before trusting that as a refutation, sanity-check the instrument
itself: does ru_minflt actually move for an allocation that SHOULD fault?
Then sweep allocation size across glibc's default mmap_threshold (128 KiB)
to see whether/where the tail (not just the mean) responds.

Pure Python, no numba kernel -- isolates the allocator effect from
everything else in score(). For each size: n calls of
    buf = bytearray(size); buf[0] = 1; buf[-1] = 1; del buf
timed individually, with ru_minflt/ru_majflt captured as an aggregate delta
over the whole n-call block (not per-call, to keep this cheap -- the
per-call rusage/cpu-migration correlation question was already answered by
tail_cause.py; this script answers a different question: does the tail
SCALE with allocation size).
"""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_JSONL = HERE / "alloc_sweep_results.jsonl"
if RESULTS_JSONL.exists():
    RESULTS_JSONL.unlink()


def jsonl_append(rec: dict) -> None:
    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()


def pctiles_ns(ns: np.ndarray) -> dict:
    return {
        "p50_us": float(np.percentile(ns, 50)) / 1e3,
        "p99_us": float(np.percentile(ns, 99)) / 1e3,
        "p99_9_us": float(np.percentile(ns, 99.9)) / 1e3,
        "max_us": float(ns.max()) / 1e3,
    }


# ---------------------------------------------------------------------------
# Sanity check: does ru_minflt move at all on this box/kernel for a
# guaranteed-fresh large allocation? (validates the instrument before we
# trust its "always zero" reading in tail_cause.py)
# ---------------------------------------------------------------------------
gc.collect()
r0 = resource.getrusage(resource.RUSAGE_SELF)
big = bytearray(64 * 1024 * 1024)  # 64 MiB, definitely fresh pages
for i in range(0, len(big), 4096):
    big[i] = 1  # touch every page to force faults
r1 = resource.getrusage(resource.RUSAGE_SELF)
print(f"sanity: touching 64MiB fresh -> minflt delta = {r1.ru_minflt - r0.ru_minflt} "
      f"(expect ~{64*1024*1024//4096}), majflt delta = {r1.ru_majflt - r0.ru_majflt}")
del big
jsonl_append({"measurement": "sanity_minflt_delta", "minflt_delta": r1.ru_minflt - r0.ru_minflt,
              "majflt_delta": r1.ru_majflt - r0.ru_majflt, "expected_pages": 64 * 1024 * 1024 // 4096})


# ---------------------------------------------------------------------------
# Sweep: allocate + touch + free a single buffer of `size` bytes, n times,
# bracketing glibc's default M_MMAP_THRESHOLD (128 KiB).
# ---------------------------------------------------------------------------
SIZES = [4 * 1024, 32 * 1024, 96 * 1024, 128 * 1024, 160 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024]
N_PER_SIZE = 3000
WARMUP = 50

summary = []
for size in SIZES:
    def one():
        buf = bytearray(size)
        buf[0] = 1
        buf[-1] = 1
        del buf

    for _ in range(WARMUP):
        one()

    gc.collect()
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    lat = np.empty(N_PER_SIZE, dtype=np.int64)
    for i in range(N_PER_SIZE):
        t0 = time.perf_counter_ns()
        one()
        t1 = time.perf_counter_ns()
        lat[i] = t1 - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)

    pt = pctiles_ns(lat)
    minflt_total = r1.ru_minflt - r0.ru_minflt
    majflt_total = r1.ru_majflt - r0.ru_majflt
    row = {
        "measurement": "alloc_sweep", "size_bytes": size, "n": N_PER_SIZE,
        **pt,
        "minflt_total": minflt_total, "majflt_total": majflt_total,
        "minflt_per_call": minflt_total / N_PER_SIZE, "majflt_per_call": majflt_total / N_PER_SIZE,
        "tail_ratio_max_over_p50": pt["max_us"] / pt["p50_us"] if pt["p50_us"] > 0 else None,
    }
    summary.append(row)
    jsonl_append(row)
    print(f"size={size:>8}B  p50={pt['p50_us']:.2f}us  p99.9={pt['p99_9_us']:.2f}us  "
          f"max={pt['max_us']:.2f}us  minflt/call={row['minflt_per_call']:.3f}  "
          f"majflt/call={row['majflt_per_call']:.4f}")

(HERE / "alloc_sweep_summary.json").write_text(json.dumps(summary, indent=2))
print("DONE")
