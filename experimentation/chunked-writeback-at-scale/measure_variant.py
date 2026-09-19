"""
EXPERIMENT J2 -- worker process. Runs exactly ONE measurement (one variant at
one row count, unchunked; or one variant at one chunk size, chunked over a
total) and exits. Always launched as its own OS process by run_j2.py via
subprocess.run(), never imported -- so a memory spike, a MemoryError, or a
kill takes only this process, and its own peak RSS (via
resource.getrusage(RUSAGE_SELF).ru_maxrss, Linux high-water mark in KB) is a
clean, isolated number instead of one contaminated by whatever else the
orchestrator or a sibling measurement was holding. This is what makes
"never hold more than one variant live" (MEASURE rule 2) true by
construction rather than by discipline within one process.

Two modes:
  --mode peak     one variant, one row count, single shot (no chunking).
                   Used for MEASURE 1 (peak RSS vs row count -> fit -> largest
                   batch that fits a 6 GB cap).
  --mode chunked  one variant, --total rows processed in chunks of
                   --chunk-size, each chunk's result WRITTEN TO A PARQUET
                   PART FILE as it completes (the "append to the result"
                   step) rather than concatenated into one in-memory frame.
                   A full in-memory concat of N/C chunk frames would total
                   the SAME bytes as the unchunked case and defeat the whole
                   point (that is a smaller-scale replay of exactly what
                   OOM-killed the box in experiment J: holding full-size
                   results live). Writing each chunk's frame to disk and
                   freeing it is the design that makes peak RSS bounded by
                   chunk size instead of total rows -- MEASURE 3's question.
                   Used for MEASURE 2/3/4.

Every measurement (mode=peak call, or mode=chunked call) appends exactly one
summary JSON line to --out, flushed immediately. --mode chunked additionally
appends one small progress line per chunk to --log (rows done so far, peak
RSS so far) so a kill mid-run still leaves a data point, per MEASURE rule 3
/ the harness's incremental-writeback rule.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shapes as S  # noqa: E402


def peak_rss_kb() -> int:
    # Linux: ru_maxrss is KB and is a lifetime HIGH-WATER MARK for this
    # process, not current usage -- exactly what "peak RSS" means here.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def make_out_arrays(variant: str, n: int):
    if variant == "record":
        return np.zeros(n, dtype=S.OUT_REC_DTYPE)
    row_major = variant == "row_major"
    if row_major:
        return (
            np.empty((n, S.OUT_F8), dtype=np.float64),
            np.empty((n, S.OUT_I8), dtype=np.int64),
            np.empty((n, S.OUT_BOOL), dtype=np.bool_),
        )
    return (
        np.empty((S.OUT_F8, n), dtype=np.float64),
        np.empty((S.OUT_I8, n), dtype=np.int64),
        np.empty((S.OUT_BOOL, n), dtype=np.bool_),
    )


def call_driver(driver, variant: str, in_rec, coef_f8, out):
    if variant == "record":
        driver(in_rec, coef_f8, out)
    else:
        out_f8, out_i8, out_b = out
        driver(in_rec, coef_f8, out_f8, out_i8, out_b)


def writeback(variant: str, out) -> "object":
    if variant == "record":
        return S.writeback_record(out)
    out_f8, out_i8, out_b = out
    return S.writeback_2d(out_f8, out_i8, out_b, row_major=(variant == "row_major"))


def checksum(frame) -> dict:
    """Cheap, deterministic fingerprint of one chunk's output -- compared
    OFFLINE, across variant runs, from the small jsonl only. Never requires
    both variants' frames resident at once (MEASURE rule 2)."""
    return {
        "f8_0_sum": float(frame[S.OUT_F8_NAMES[0]].sum()),
        "f8_last_sum": float(frame[S.OUT_F8_NAMES[-1]].sum()),
        "i8_0_sum": int(frame[S.OUT_I8_NAMES[0]].sum()),
        "b_0_true_count": int(frame[S.OUT_B_NAMES[0]].sum()),
        "n": frame.height,
    }


def compile_driver(driver, variant: str, coef_f8):
    """Warm/trigger compile on a tiny array (n=2), independent of the real
    row count or chunk size -- numba's signature cache is keyed on dtype,
    not size, so a 2-row warm call is enough to pay (or skip, on a cache
    HIT) the compile cost exactly once."""
    warm_rec = np.zeros(2, dtype=S.IN_DTYPE)
    warm_out = make_out_arrays(variant, 2)
    t0 = time.perf_counter()
    call_driver(driver, variant, warm_rec, coef_f8, warm_out)
    return time.perf_counter() - t0


def run_peak(args):
    n = args.n
    variant = args.variant
    est = S.estimate_peak_bytes(variant, n)
    print(f"[estimate] variant={variant} n={n:,} -> {est/1e9:.3f} GB analytic "
          f"(excludes ~150-250MB interpreter/numba/polars baseline)", flush=True)

    frame = S.make_input_frame(n, seed=7)
    cols = S.extract(frame)
    del frame
    gc.collect()
    in_rec = S.assemble(cols, n)
    del cols
    gc.collect()

    driver, path = S.load_driver(variant)
    coef_f8 = np.array([1.0 + 0.001 * j for j in range(S.OUT_F8)], dtype=np.float64)
    compile_s = compile_driver(driver, variant, coef_f8)

    repeats = args.repeats
    kernel_times = []
    wb_times = []
    frame_out = None
    for i in range(repeats + 1):  # first is warm/discarded
        out = make_out_arrays(variant, n)
        t0 = time.perf_counter()
        call_driver(driver, variant, in_rec, coef_f8, out)
        dt = time.perf_counter() - t0
        t0 = time.perf_counter()
        frame_out = writeback(variant, out)
        wdt = time.perf_counter() - t0
        if i > 0:
            kernel_times.append(dt)
            wb_times.append(wdt)
        del out

    cksum = checksum(frame_out)
    del frame_out, in_rec
    gc.collect()

    rec = {
        "kind": "peak",
        "variant": variant,
        "n": n,
        "estimate_bytes": est,
        "compile_s": compile_s,
        "kernel_ms_median": statistics.median(kernel_times) * 1e3,
        "writeback_ms_median": statistics.median(wb_times) * 1e3,
        "peak_rss_kb": peak_rss_kb(),
        "checksum": cksum,
    }
    append_jsonl(Path(args.out), rec)
    print(f"[result] {json.dumps(rec)}", flush=True)


def run_chunked(args):
    variant = args.variant
    total = args.total
    chunk_size = args.chunk_size
    out_path = Path(args.out)
    log_path = Path(args.log)
    parts_dir = Path(args.parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    est_chunk = S.estimate_peak_bytes(variant, chunk_size)
    print(f"[estimate] variant={variant} chunk_size={chunk_size:,} total={total:,} "
          f"-> {est_chunk/1e9:.3f} GB analytic peak PER CHUNK "
          f"(bounded by chunk_size, not total -- that is the property under test)",
          flush=True)

    driver, path = S.load_driver(variant)
    coef_f8 = np.array([1.0 + 0.001 * j for j in range(S.OUT_F8)], dtype=np.float64)
    # Generation-pointer discipline (EXPERIMENTS.md SH): the driver and
    # coef_f8 are obtained ONCE here, before the chunk loop, and reused for
    # every chunk -- never re-resolved per chunk.
    compile_s = compile_driver(driver, variant, coef_f8)

    n_chunks = (total + chunk_size - 1) // chunk_size
    kernel_total = 0.0
    wb_total = 0.0
    rows_done = 0
    checksums = []
    wall_t0 = time.perf_counter()

    for idx in range(n_chunks):
        n_c = min(chunk_size, total - rows_done)
        frame = S.make_input_frame(n_c, seed=1000 + idx)
        cols = S.extract(frame)
        del frame
        in_rec = S.assemble(cols, n_c)
        del cols

        out = make_out_arrays(variant, n_c)
        t0 = time.perf_counter()
        call_driver(driver, variant, in_rec, coef_f8, out)
        kernel_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        frame_out = writeback(variant, out)
        wb_total += time.perf_counter() - t0

        cksum = checksum(frame_out)
        checksums.append(cksum)

        part_path = parts_dir / f"{variant}_c{chunk_size}_part{idx:05d}.parquet"
        frame_out.write_parquet(part_path)

        rows_done += n_c
        del in_rec, out, frame_out
        gc.collect()

        append_jsonl(log_path, {
            "kind": "chunk_progress", "variant": variant, "chunk_size": chunk_size,
            "chunk_idx": idx, "n_chunks": n_chunks, "rows_done": rows_done,
            "peak_rss_kb": peak_rss_kb(), "t_s": time.perf_counter() - wall_t0,
        })

    wall_s = time.perf_counter() - wall_t0
    rec = {
        "kind": "chunked", "variant": variant, "total": total, "chunk_size": chunk_size,
        "n_chunks": n_chunks, "estimate_peak_bytes_per_chunk": est_chunk,
        "compile_s": compile_s,
        "kernel_total_ms": kernel_total * 1e3, "writeback_total_ms": wb_total * 1e3,
        "wall_s": wall_s,
        "boundary_overhead_ms": (wall_s - kernel_total - wb_total - compile_s) * 1e3,
        "ns_per_row_kernel": kernel_total * 1e9 / total,
        "ns_per_row_writeback": wb_total * 1e9 / total,
        "peak_rss_kb": peak_rss_kb(),
        "rows_done": rows_done,
        "checksums": checksums,
        "part_files": n_chunks,
    }
    append_jsonl(out_path, rec)
    print(f"[result] variant={variant} chunk_size={chunk_size} wall={wall_s:.2f}s "
          f"peak_rss={peak_rss_kb()/1e6:.3f}GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["peak", "chunked"], required=True)
    ap.add_argument("--variant", choices=["record", "col_major", "row_major"], required=True)
    ap.add_argument("--n", type=int, default=None, help="peak mode: row count")
    ap.add_argument("--total", type=int, default=None, help="chunked mode: total rows")
    ap.add_argument("--chunk-size", type=int, default=None, help="chunked mode")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None, help="chunked mode: per-chunk progress jsonl")
    ap.add_argument("--parts-dir", default=None, help="chunked mode: where chunk parquet parts go")
    args = ap.parse_args()

    if args.mode == "peak":
        run_peak(args)
    else:
        run_chunked(args)


if __name__ == "__main__":
    main()
