"""
EXPERIMENT J2 -- orchestrator. THIS SCRIPT NEVER ALLOCATES THE ARRAYS ITSELF.
Every measurement is a subprocess of measure_variant.py, run to completion,
its one result line already flushed to disk, before the next one starts.
That is what "never hold more than one variant live" and "a kill costs one
data point" mean in practice here -- not just del+gc.collect() inside a
single process (which is what experiment J's writeback.py did NOT do, and
which is diagnosed in README.md).

Run (from inside this directory, under the mandatory tmux+systemd-run wrapper
-- see README.md):
    <repo>/.venv/bin/python run_j2.py

Writes:
    results_peak.jsonl      -- one line per (variant, row count), MEASURE 1
    results_chunked.jsonl   -- one line per (variant, chunk size), MEASURE 2/3/4
    chunk_progress.jsonl    -- one line per chunk processed (all configs)
    summary.json            -- fit + recommendation, written at the very end
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shapes as S  # noqa: E402

HERE = Path(__file__).resolve().parent
PY = sys.executable
WORKER = HERE / "measure_variant.py"

RESULTS_PEAK = HERE / "results_peak.jsonl"
RESULTS_CHUNKED = HERE / "results_chunked.jsonl"
CHUNK_PROGRESS = HERE / "chunk_progress.jsonl"
PARTS_DIR = HERE / "_parts"

CAP_BYTES = 4 * 1024 ** 3       # MEASURE rule 1: restructure rather than run above this
SYSTEMD_CAP_BYTES = 6 * 1024 ** 3  # matches -p MemoryMax=6G in the launch wrapper

PEAK_ROW_COUNTS = [10_000, 50_000, 100_000, 250_000]
PEAK_VARIANTS = ["record", "col_major"]

CHUNK_SIZES = [10_000, 50_000, 100_000, 250_000]
CHUNK_VARIANTS = ["record", "col_major"]
CHUNK_TOTAL = 1_000_000

T0 = time.perf_counter()


def elapsed() -> float:
    return time.perf_counter() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


def check_free_memory_or_die():
    out = subprocess.run(["free", "-g"], capture_output=True, text=True, check=True).stdout
    log("free -g:\n" + out.rstrip())
    mem_line = [l for l in out.splitlines() if l.startswith("Mem:")][0]
    parts = mem_line.split()
    available_gb = int(parts[6]) if len(parts) > 6 else int(parts[3])
    if available_gb < 8:
        log(f"ABORT: available memory {available_gb} GB < 8 GB minimum. Not launching (rule 5).")
        sys.exit(1)
    log(f"available memory {available_gb} GB >= 8 GB -- proceeding.")


def run_worker(argv: list[str]) -> None:
    cmd = [PY, str(WORKER)] + argv
    log("subprocess: " + " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(HERE))
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        log(f"WORKER EXIT {proc.returncode} after {dt:.1f}s -- treated as a DATA POINT, "
            f"not a fatal error (may be a 137/MemoryError under the cap). Continuing.")
    else:
        log(f"worker done in {dt:.1f}s")


def part1_peak_sweep():
    log("=== PART 1: peak RSS per row count, one variant at a time ===")
    for variant in PEAK_VARIANTS:
        for n in PEAK_ROW_COUNTS:
            est = S.estimate_peak_bytes(variant, n)
            log(f"[pre-run estimate] variant={variant} n={n:,} -> {est/1e9:.3f} GB analytic")
            if est > CAP_BYTES:
                log(f"SKIP: estimate {est/1e9:.2f} GB exceeds the {CAP_BYTES/1e9:.0f} GB "
                    f"restructure threshold (MEASURE rule 1). Not running.")
                continue
            run_worker([
                "--mode", "peak", "--variant", variant, "--n", str(n),
                "--repeats", "3", "--out", str(RESULTS_PEAK),
            ])


def linfit(xs, ys):
    """Least-squares y = a*x + b, no numpy needed for 4 points."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx else 0.0
    b = my - a * mx
    return a, b


def part1_fit_and_report():
    if not RESULTS_PEAK.exists():
        log("no peak results to fit.")
        return {}
    rows = [json.loads(l) for l in RESULTS_PEAK.read_text().splitlines() if l.strip()]
    fits = {}
    for variant in PEAK_VARIANTS:
        pts = sorted((r["n"], r["peak_rss_kb"] * 1024) for r in rows if r["variant"] == variant)
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        a, b = linfit(xs, ys)
        pred_1m = a * 1_000_000 + b
        n_at_cap = (SYSTEMD_CAP_BYTES - b) / a if a > 0 else float("inf")
        fits[variant] = {
            "points_n_bytes": pts,
            "fit_bytes_per_row": a,
            "fit_intercept_bytes": b,
            "predicted_bytes_at_1e6": pred_1m,
            "predicted_GB_at_1e6": pred_1m / 1e9,
            "largest_n_under_6GB_cap": int(n_at_cap),
        }
        log(f"[fit] {variant}: {a:.1f} B/row + {b/1e6:.1f} MB baseline -> "
            f"predicted {pred_1m/1e9:.2f} GB at 1e6 rows; "
            f"largest batch under 6GB cap ~= {int(n_at_cap):,} rows")
    return fits


def part2_chunked_sweep():
    log("=== PART 2/3: chunked processing, 1M rows total, sweeping chunk size ===")
    for variant in CHUNK_VARIANTS:
        for c in CHUNK_SIZES:
            est = S.estimate_peak_bytes(variant, c)
            log(f"[pre-run estimate] variant={variant} chunk_size={c:,} total={CHUNK_TOTAL:,} "
                f"-> {est/1e9:.3f} GB analytic PEAK PER CHUNK (not total)")
            if est > CAP_BYTES:
                log(f"SKIP: per-chunk estimate {est/1e9:.2f} GB exceeds "
                    f"{CAP_BYTES/1e9:.0f} GB threshold. Not running.")
                continue
            run_worker([
                "--mode", "chunked", "--variant", variant,
                "--total", str(CHUNK_TOTAL), "--chunk-size", str(c),
                "--out", str(RESULTS_CHUNKED), "--log", str(CHUNK_PROGRESS),
                "--parts-dir", str(PARTS_DIR),
            ])


def part2_report_and_checksum_crosscheck():
    if not RESULTS_CHUNKED.exists():
        log("no chunked results.")
        return {}
    rows = [json.loads(l) for l in RESULTS_CHUNKED.read_text().splitlines() if l.strip()]
    by_key = {}
    for r in rows:
        by_key.setdefault(r["chunk_size"], {})[r["variant"]] = r

    mismatches = 0
    compared = 0
    for c, variants in by_key.items():
        if "record" not in variants or "col_major" not in variants:
            continue
        rec_cks = variants["record"]["checksums"]
        col_cks = variants["col_major"]["checksums"]
        for i, (a, b) in enumerate(zip(rec_cks, col_cks)):
            compared += 1
            ok = (
                abs(a["f8_0_sum"] - b["f8_0_sum"]) < 1e-6
                and abs(a["f8_last_sum"] - b["f8_last_sum"]) < 1e-6
                and a["i8_0_sum"] == b["i8_0_sum"]
                and a["b_0_true_count"] == b["b_0_true_count"]
                and a["n"] == b["n"]
            )
            if not ok:
                mismatches += 1
                log(f"CHECKSUM MISMATCH chunk_size={c} chunk_idx={i}: record={a} col_major={b}")
    log(f"[crosscheck] {compared} chunk checksums compared across record vs col_major "
        f"(NEVER holding both variants' frames resident -- compared from jsonl only), "
        f"{mismatches} mismatches")

    summary_by_config = []
    for r in rows:
        row_share = 100.0 * r["kernel_total_ms"] / (r["wall_s"] * 1e3)
        wb_share = 100.0 * r["writeback_total_ms"] / (r["wall_s"] * 1e3)
        bnd_share = 100.0 * r["boundary_overhead_ms"] / (r["wall_s"] * 1e3)
        summary_by_config.append({
            "variant": r["variant"], "chunk_size": r["chunk_size"],
            "n_chunks": r["n_chunks"], "wall_s": r["wall_s"],
            "kernel_pct": row_share, "writeback_pct": wb_share, "boundary_pct": bnd_share,
            "peak_rss_GB": r["peak_rss_kb"] / 1e6,
            "ns_per_row_kernel": r["ns_per_row_kernel"],
        })
        log(f"[chunked result] variant={r['variant']:10s} chunk={r['chunk_size']:>7,} "
            f"n_chunks={r['n_chunks']:>3} wall={r['wall_s']:6.2f}s "
            f"kernel={row_share:5.1f}% wb={wb_share:5.1f}% boundary={bnd_share:5.1f}% "
            f"peak_rss={r['peak_rss_kb']/1e6:5.2f}GB")
    return {"checksum_compared": compared, "checksum_mismatches": mismatches,
            "by_config": summary_by_config}


def main():
    check_free_memory_or_die()
    PARTS_DIR.mkdir(exist_ok=True)

    part1_peak_sweep()
    fits = part1_fit_and_report()

    part2_chunked_sweep()
    chunked_summary = part2_report_and_checksum_crosscheck()

    summary = {"peak_fits": fits, "chunked": chunked_summary, "elapsed_s": elapsed()}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"DONE. total elapsed {elapsed():.1f}s. wrote summary.json")


if __name__ == "__main__":
    main()
