"""
EXPERIMENT J -- the output write-back convention (doc 01 S4d / doc 05 S3).

E9 settled "1D structured (record) array per side" and named its own
bottleneck: "Record write-back becomes the new bottleneck -- gathering
strided fields back out costs 35-158 ms, which is 54.7% of total time at
633 outputs." This harness tests the untested review proposal: keep the
INPUT side a record array, make the OUTPUT side dtype-grouped 2D arrays,
column-major (so out_f8[j, :] is a contiguous slice) -- against a
row-major control that isolates "2D instead of records" from "column-major
instead of row-major".

Three conventions, all fed by the SAME assembled input record array and
computing the IDENTICAL per-output arithmetic (see gen_*_source below), so
kernel-time differences come only from the write pattern, not the formula:

  1. record in  -> record out          (current spec, doc 05 S3)
  2. record in  -> dtype-grouped COLUMN-major 2D out   (untested proposal)
  3. record in  -> dtype-grouped ROW-major 2D out       (control)

Reused from prior harnesses in experimentation/ (not rewritten):
  - elapsed()/log() progress-clock pattern       <- prange-crossover/prange_crossover.py
  - median-of-repeats timing (warm call, then time `repeats` calls)
                                                  <- dtype-boundary/dtype_boundary.py:median_us
  - write generated source to a REAL .py file and import it via
    importlib.util.spec_from_file_location (never exec()), per doc 05 S4.1
                                                  <- ruleset-compile-latency/run.py:load_source
  - zero-copy check via Series._get_buffer_info()[0] vs
    arr.__array_interface__["data"][0]           <- dtype-boundary/dtype_boundary.py:probe_to_numpy
  - extraction via Series._get_buffers()["values"].to_numpy() (no pyarrow)
                                                  <- dtype-boundary/dtype_boundary.py (per EXPERIMENTS.md SA)

Run:
    /path/to/.venv/bin/python writeback.py                     # full run (100k, 1M)
    /path/to/.venv/bin/python writeback.py --sizes 100000       # just 100k
    /path/to/.venv/bin/python writeback.py --quick               # small smoke sweep

Writes results.json next to itself.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import numba
import polars as pl

HERE = Path(__file__).resolve().parent
T0 = time.perf_counter()


def elapsed() -> float:
    return time.perf_counter() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# Shape: 400 inputs / 633 outputs, mixed f8/i8/bool (doc 01 S4d's verified
# target shape). Split chosen to look like real decision-engine output
# (mostly numeric, some integer codes, some flags); held IDENTICAL across
# all three conventions and stated here, not tuned per result.
# --------------------------------------------------------------------------
IN_F8, IN_I8, IN_BOOL = 300, 80, 20
N_IN = IN_F8 + IN_I8 + IN_BOOL
assert N_IN == 400

OUT_F8, OUT_I8, OUT_BOOL = 380, 158, 95
N_OUT = OUT_F8 + OUT_I8 + OUT_BOOL
assert N_OUT == 633

IN_F8_NAMES = [f"i_f8_{k}" for k in range(IN_F8)]
IN_I8_NAMES = [f"i_i8_{k}" for k in range(IN_I8)]
IN_B_NAMES = [f"i_b_{k}" for k in range(IN_BOOL)]
IN_ALL_NAMES = IN_F8_NAMES + IN_I8_NAMES + IN_B_NAMES

OUT_F8_NAMES = [f"o_f8_{k}" for k in range(OUT_F8)]
OUT_I8_NAMES = [f"o_i8_{k}" for k in range(OUT_I8)]
OUT_B_NAMES = [f"o_b_{k}" for k in range(OUT_BOOL)]
OUT_ALL_NAMES = OUT_F8_NAMES + OUT_I8_NAMES + OUT_B_NAMES

IN_DTYPE = np.dtype(
    [(n, "f8") for n in IN_F8_NAMES]
    + [(n, "i8") for n in IN_I8_NAMES]
    + [(n, "?") for n in IN_B_NAMES]
)
OUT_REC_DTYPE = np.dtype(
    [(n, "f8") for n in OUT_F8_NAMES]
    + [(n, "i8") for n in OUT_I8_NAMES]
    + [(n, "?") for n in OUT_B_NAMES]
)


# --------------------------------------------------------------------------
# Timing helper -- reused pattern from dtype-boundary/dtype_boundary.py's
# median_us: warm once (compiles / faults in pages), then time `repeats`
# calls, report the median. Returns (median_seconds, all_samples_seconds).
# --------------------------------------------------------------------------
def median_s(fn, repeats: int):
    fn()  # warm
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


# --------------------------------------------------------------------------
# Codegen -- real files, never exec(), per doc 05 S4.1 / reused pattern from
# ruleset-compile-latency/run.py:load_source.
# --------------------------------------------------------------------------
_GEN_DIR = HERE / "_generated"
_GEN_DIR.mkdir(exist_ok=True)
_UID = [0]


def load_source(src: str, fname: str = "driver"):
    _UID[0] += 1
    path = _GEN_DIR / f"gen_{_UID[0]}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"gen_{_UID[0]}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fname), path


# Shared row-level taps (same for all three variants): a small fixed set of
# input fields read once per row, then reused across all 633 outputs. This
# keeps the ARITHMETIC identical across conventions -- only the write
# target differs -- which is what makes the kernel-only comparison mean
# "cost of the write pattern" rather than "cost of a different formula".
_HEADER = """\
import numpy as np
from numba import njit


@njit(cache=False, nogil=True)
def driver(inp, coef_f8, {out_args}):
    n = inp.shape[0]
    for i in range(n):
        a0 = inp[i].i_f8_0
        a1 = inp[i].i_f8_1
        a2 = inp[i].i_f8_2
        a3 = inp[i].i_f8_3
        a4 = inp[i].i_f8_4
        b0 = inp[i].i_i8_0
"""


def gen_record_out_source() -> str:
    lines = [_HEADER.format(out_args="out")]
    for j, name in enumerate(OUT_F8_NAMES):
        lines.append(f"        out[i].{name} = coef_f8[{j}] * a0 + a1 - a2 * 0.001")
    for j, name in enumerate(OUT_I8_NAMES):
        lines.append(f"        out[i].{name} = b0 + {j}")
    for j, name in enumerate(OUT_B_NAMES):
        lines.append(f"        out[i].{name} = (a3 + {j} * 0.0001) > a4")
    return "\n".join(lines) + "\n"


def gen_2d_source(row_major: bool) -> str:
    lines = [_HEADER.format(out_args="out_f8, out_i8, out_b")]
    if row_major:
        f8_lhs, i8_lhs, b_lhs = "out_f8[i, j]", "out_i8[i, j]", "out_b[i, j]"
    else:
        f8_lhs, i8_lhs, b_lhs = "out_f8[j, i]", "out_i8[j, i]", "out_b[j, i]"
    lines.append(f"        for j in range({OUT_F8}):")
    lines.append(f"            {f8_lhs} = coef_f8[j] * a0 + a1 - a2 * 0.001")
    lines.append(f"        for j in range({OUT_I8}):")
    lines.append(f"            {i8_lhs} = b0 + j")
    lines.append(f"        for j in range({OUT_BOOL}):")
    lines.append(f"            {b_lhs} = (a3 + j * 0.0001) > a4")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Synthetic input frame + extract/assemble (shared by all three conventions
# -- the input side is a record array in every variant, so these phases do
# not vary between them and are measured ONCE per row count).
# --------------------------------------------------------------------------
def make_input_frame(n: int, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    for name in IN_F8_NAMES:
        cols[name] = rng.random(n)
    for name in IN_I8_NAMES:
        cols[name] = rng.integers(-1000, 1000, n, dtype=np.int64)
    for name in IN_B_NAMES:
        cols[name] = rng.random(n) < 0.5
    return pl.DataFrame(cols)


def extract(frame: pl.DataFrame) -> dict:
    """Per-column, via polars-native buffers -- no pyarrow (EXPERIMENTS.md SA)."""
    return {name: frame[name]._get_buffers()["values"].to_numpy() for name in IN_ALL_NAMES}


def assemble(cols: dict, n: int) -> np.ndarray:
    rec = np.empty(n, dtype=IN_DTYPE)
    for name in IN_ALL_NAMES:
        rec[name] = cols[name]
    return rec


# --------------------------------------------------------------------------
# Write-back: dict of {name: array} -> pl.DataFrame(dict), per doc 05 S3's
# own recommendation ("hstack(pl.DataFrame(dict)) rather than chained
# with_columns"). We measure exactly this constructor call.
# --------------------------------------------------------------------------
def writeback_record(out_rec: np.ndarray) -> pl.DataFrame:
    d = {name: out_rec[name] for name in OUT_ALL_NAMES}
    return pl.DataFrame(d)


def writeback_2d(out_f8: np.ndarray, out_i8: np.ndarray, out_b: np.ndarray, row_major: bool) -> pl.DataFrame:
    d = {}
    if row_major:
        for j, name in enumerate(OUT_F8_NAMES):
            d[name] = out_f8[:, j]
        for j, name in enumerate(OUT_I8_NAMES):
            d[name] = out_i8[:, j]
        for j, name in enumerate(OUT_B_NAMES):
            d[name] = out_b[:, j]
    else:
        for j, name in enumerate(OUT_F8_NAMES):
            d[name] = out_f8[j, :]
        for j, name in enumerate(OUT_I8_NAMES):
            d[name] = out_i8[j, :]
        for j, name in enumerate(OUT_B_NAMES):
            d[name] = out_b[j, :]
    return pl.DataFrame(d)


def zero_copy_check(frame: pl.DataFrame, arrays: dict) -> dict:
    """For each name -> source array, does frame[name] share the numpy
    pointer (True) or did polars copy (False)? Pattern reused from
    dtype-boundary/dtype_boundary.py:probe_to_numpy's shares_address check."""
    out = {}
    for name, arr in arrays.items():
        try:
            src_ptr = arr.__array_interface__["data"][0]
            dst_ptr = frame[name]._get_buffer_info()[0]
            out[name] = bool(src_ptr == dst_ptr)
        except BaseException as e:
            out[name] = f"ERR:{type(e).__name__}"
    return out


# --------------------------------------------------------------------------
# Main experiment for one row count.
# --------------------------------------------------------------------------
def run_size(n: int, repeats: int, kernel_repeats: int, wb_repeats: int) -> dict:
    log(f"=== n={n:,} ===")
    result = {"n": n}

    frame = make_input_frame(n)
    gc.collect()

    t, samples = median_s(lambda: extract(frame), repeats)
    log(f"extract: median {t*1e3:.3f} ms over {repeats} reps  (shared by all 3 conventions)")
    result["extract_ms"] = {"median": t * 1e3, "repeats": repeats, "samples_ms": [s * 1e3 for s in samples]}

    cols = extract(frame)

    t, samples = median_s(lambda: assemble(cols, n), repeats)
    log(f"assemble: median {t*1e3:.3f} ms over {repeats} reps  (shared by all 3 conventions)")
    result["assemble_ms"] = {"median": t * 1e3, "repeats": repeats, "samples_ms": [s * 1e3 for s in samples]}

    in_rec = assemble(cols, n)
    coef_f8 = np.array([1.0 + 0.001 * j for j in range(OUT_F8)], dtype=np.float64)

    # small array for warmup (triggers compile once per kernel; the compiled
    # signature is shape-generic, so this serves both 100k and 1M runs)
    warm_rec = in_rec[: min(1000, n)].copy()

    variants = {}

    # --- 1. record in / record out (current spec) -------------------------
    log("compiling record-out driver...")
    t0 = time.perf_counter()
    drv_rec, path_rec = load_source(gen_record_out_source())
    warm_out = np.zeros(warm_rec.shape[0], dtype=OUT_REC_DTYPE)
    drv_rec(warm_rec, coef_f8, warm_out)  # trigger compile
    compile_s = time.perf_counter() - t0
    log(f"  record-out compiled in {compile_s:.2f}s -> {path_rec.name}")

    def run_rec():
        out_rec = np.zeros(n, dtype=OUT_REC_DTYPE)
        drv_rec(in_rec, coef_f8, out_rec)
        return out_rec

    kt, ksamples = median_s(run_rec, kernel_repeats)
    out_rec_final = run_rec()
    wbt, wbsamples = median_s(lambda: writeback_record(out_rec_final), wb_repeats)
    frame_rec = writeback_record(out_rec_final)
    variants["record"] = {
        "compile_s": compile_s,
        "kernel_ms": {"median": kt * 1e3, "repeats": kernel_repeats, "samples_ms": [s * 1e3 for s in ksamples]},
        "writeback_ms": {"median": wbt * 1e3, "repeats": wb_repeats, "samples_ms": [s * 1e3 for s in wbsamples]},
        "ns_per_row_kernel": kt * 1e9 / n,
    }
    log(f"  record: kernel median {kt*1e3:.2f} ms, write-back median {wbt*1e3:.2f} ms")

    # --- 2. record in / dtype-grouped COLUMN-major 2D out (proposal) ------
    log("compiling column-major 2D-out driver...")
    t0 = time.perf_counter()
    drv_col, path_col = load_source(gen_2d_source(row_major=False))
    wf8 = np.zeros((OUT_F8, warm_rec.shape[0]), dtype=np.float64)
    wi8 = np.zeros((OUT_I8, warm_rec.shape[0]), dtype=np.int64)
    wb_ = np.zeros((OUT_BOOL, warm_rec.shape[0]), dtype=np.bool_)
    drv_col(warm_rec, coef_f8, wf8, wi8, wb_)
    compile_s = time.perf_counter() - t0
    log(f"  col-major compiled in {compile_s:.2f}s -> {path_col.name}")

    def run_col():
        out_f8 = np.empty((OUT_F8, n), dtype=np.float64)
        out_i8 = np.empty((OUT_I8, n), dtype=np.int64)
        out_b = np.empty((OUT_BOOL, n), dtype=np.bool_)
        drv_col(in_rec, coef_f8, out_f8, out_i8, out_b)
        return out_f8, out_i8, out_b

    kt, ksamples = median_s(run_col, kernel_repeats)
    cf8, ci8, cb = run_col()
    wbt, wbsamples = median_s(lambda: writeback_2d(cf8, ci8, cb, row_major=False), wb_repeats)
    frame_col = writeback_2d(cf8, ci8, cb, row_major=False)
    variants["col_major_2d"] = {
        "compile_s": compile_s,
        "kernel_ms": {"median": kt * 1e3, "repeats": kernel_repeats, "samples_ms": [s * 1e3 for s in ksamples]},
        "writeback_ms": {"median": wbt * 1e3, "repeats": wb_repeats, "samples_ms": [s * 1e3 for s in wbsamples]},
        "ns_per_row_kernel": kt * 1e9 / n,
    }
    log(f"  col-major 2D: kernel median {kt*1e3:.2f} ms, write-back median {wbt*1e3:.2f} ms")

    # zero-copy check on the column-major write-back (a fresh construction,
    # not reusing frame_col, so the pointers checked are exactly the ones
    # that produced the frame we verify below)
    check_arrays = {}
    for j, name in enumerate(OUT_F8_NAMES[:5] + OUT_F8_NAMES[-5:]):
        idx = OUT_F8_NAMES.index(name)
        check_arrays[name] = cf8[idx, :]
    zc = zero_copy_check(frame_col, check_arrays)
    variants["col_major_2d"]["zero_copy_sample"] = zc
    log(f"  col-major zero-copy (10-col sample): {sum(1 for v in zc.values() if v is True)}/10 shared pointer")

    # --- 3. record in / dtype-grouped ROW-major 2D out (control) ----------
    log("compiling row-major 2D-out driver...")
    t0 = time.perf_counter()
    drv_row, path_row = load_source(gen_2d_source(row_major=True))
    wf8r = np.zeros((warm_rec.shape[0], OUT_F8), dtype=np.float64)
    wi8r = np.zeros((warm_rec.shape[0], OUT_I8), dtype=np.int64)
    wbr_ = np.zeros((warm_rec.shape[0], OUT_BOOL), dtype=np.bool_)
    drv_row(warm_rec, coef_f8, wf8r, wi8r, wbr_)
    compile_s = time.perf_counter() - t0
    log(f"  row-major compiled in {compile_s:.2f}s -> {path_row.name}")

    def run_row():
        out_f8 = np.empty((n, OUT_F8), dtype=np.float64)
        out_i8 = np.empty((n, OUT_I8), dtype=np.int64)
        out_b = np.empty((n, OUT_BOOL), dtype=np.bool_)
        drv_row(in_rec, coef_f8, out_f8, out_i8, out_b)
        return out_f8, out_i8, out_b

    kt, ksamples = median_s(run_row, kernel_repeats)
    rf8, ri8, rb = run_row()
    wbt, wbsamples = median_s(lambda: writeback_2d(rf8, ri8, rb, row_major=True), wb_repeats)
    frame_row = writeback_2d(rf8, ri8, rb, row_major=True)
    variants["row_major_2d"] = {
        "compile_s": compile_s,
        "kernel_ms": {"median": kt * 1e3, "repeats": kernel_repeats, "samples_ms": [s * 1e3 for s in ksamples]},
        "writeback_ms": {"median": wbt * 1e3, "repeats": wb_repeats, "samples_ms": [s * 1e3 for s in wbsamples]},
        "ns_per_row_kernel": kt * 1e9 / n,
    }
    log(f"  row-major 2D: kernel median {kt*1e3:.2f} ms, write-back median {wbt*1e3:.2f} ms")

    check_arrays_row = {}
    for name in OUT_F8_NAMES[:5] + OUT_F8_NAMES[-5:]:
        idx = OUT_F8_NAMES.index(name)
        check_arrays_row[name] = rf8[:, idx]
    zcr = zero_copy_check(frame_row, check_arrays_row)
    variants["row_major_2d"]["zero_copy_sample"] = zcr
    log(f"  row-major zero-copy (10-col sample): {sum(1 for v in zcr.values() if v is True)}/10 shared pointer")

    # --- 4. verify all three produce identical output frames --------------
    ident_f8 = np.allclose(
        frame_rec.select(OUT_F8_NAMES).to_numpy(),
        frame_col.select(OUT_F8_NAMES).to_numpy(),
    ) and np.allclose(
        frame_rec.select(OUT_F8_NAMES).to_numpy(),
        frame_row.select(OUT_F8_NAMES).to_numpy(),
    )
    ident_i8 = np.array_equal(
        frame_rec.select(OUT_I8_NAMES).to_numpy(), frame_col.select(OUT_I8_NAMES).to_numpy()
    ) and np.array_equal(
        frame_rec.select(OUT_I8_NAMES).to_numpy(), frame_row.select(OUT_I8_NAMES).to_numpy()
    )
    ident_b = np.array_equal(
        frame_rec.select(OUT_B_NAMES).to_numpy(), frame_col.select(OUT_B_NAMES).to_numpy()
    ) and np.array_equal(
        frame_rec.select(OUT_B_NAMES).to_numpy(), frame_row.select(OUT_B_NAMES).to_numpy()
    )
    result["identical_output_frames"] = {"f8": bool(ident_f8), "i8": bool(ident_i8), "bool": bool(ident_b)}
    log(f"  identical output frames: f8={ident_f8} i8={ident_i8} bool={ident_b}")

    # --- totals + percentage breakdown (end-to-end: extract+assemble shared,
    # + per-variant kernel + write-back) -----------------------------------
    shared_ms = result["extract_ms"]["median"] + result["assemble_ms"]["median"]
    for vname, v in variants.items():
        total_ms = shared_ms + v["kernel_ms"]["median"] + v["writeback_ms"]["median"]
        v["total_ms"] = total_ms
        v["writeback_pct_of_total"] = 100.0 * v["writeback_ms"]["median"] / total_ms

    result["shared_ms"] = shared_ms
    result["variants"] = variants

    del frame, cols, in_rec, out_rec_final, cf8, ci8, cb, rf8, ri8, rb
    del frame_rec, frame_col, frame_row
    gc.collect()

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100000,1000000")
    ap.add_argument("--repeats", type=int, default=5, help="extract/assemble repeats")
    ap.add_argument("--kernel-repeats", type=int, default=7)
    ap.add_argument("--wb-repeats", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        sizes = [2000]
        repeats, kernel_repeats, wb_repeats = 3, 3, 3
    else:
        sizes = [int(s) for s in args.sizes.split(",") if s]
        repeats, kernel_repeats, wb_repeats = args.repeats, args.kernel_repeats, args.wb_repeats

    log(f"env: python={platform.python_version()} numba={numba.__version__} "
        f"numpy={np.__version__} polars={pl.__version__}")
    log(f"shape: {N_IN} in ({IN_F8}f8/{IN_I8}i8/{IN_BOOL}bool), "
        f"{N_OUT} out ({OUT_F8}f8/{OUT_I8}i8/{OUT_BOOL}bool)")
    log(f"sizes={sizes} repeats(extract/assemble)={repeats} "
        f"kernel_repeats={kernel_repeats} wb_repeats={wb_repeats}")

    results = {
        "env": {
            "python": platform.python_version(),
            "numba": numba.__version__,
            "numpy": np.__version__,
            "polars": pl.__version__,
        },
        "shape": {
            "n_in": N_IN, "in_f8": IN_F8, "in_i8": IN_I8, "in_bool": IN_BOOL,
            "n_out": N_OUT, "out_f8": OUT_F8, "out_i8": OUT_I8, "out_bool": OUT_BOOL,
        },
        "by_size": [],
    }

    for n in sizes:
        r = run_size(n, repeats, kernel_repeats, wb_repeats)
        results["by_size"].append(r)
        (HERE / "results.json").write_text(json.dumps(results, indent=2, default=str))
        log(f"wrote results.json after n={n:,}  (total elapsed {elapsed():.1f}s)")

    log("DONE")


if __name__ == "__main__":
    main()
