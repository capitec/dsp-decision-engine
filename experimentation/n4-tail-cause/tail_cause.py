"""
Chasing N4's unexplained tail (EXPERIMENTS.md SS N4/M1: p99.9 1723us / 8.6%
of a 20ms budget, max 3840us / 19.2%, GC ruled out, "not isolated further").

Reuses experimentation/single-record-overhead/n1_overhead.py's score() /
DRIVER / REQUEST_DICT / PARAMS_RAW / rss_mb / pctiles verbatim, imported
directly (its heavy sweep code is behind `if __name__ == "__main__":`, so
importing only pays the one-time kernel compile + a sanity call).

Per call, capture latency_ns AND four independent signals, then correlate
the slow calls against each signal (not a theory):
  - resource.getrusage(RUSAGE_SELF) deltas: ru_minflt/ru_majflt (page
    faults), ru_nvcsw/ru_nivcsw (voluntary/involuntary context switches)
  - current CPU core. os.sched_getcpu() does NOT exist on this Python 3.14
    build (checked: AttributeError) -- read instead from /proc/self/stat
    field 39 ("processor", proc(5)) via a held-open fd + lseek(0)+read,
    measured at 2.6us/call vs 5.4us for open+read+close.
  - sys.getallocatedblocks() delta (CPython allocator block count)
  - gc.get_count()[0] delta (cheap proxy; GC collection frequency itself
    already ruled out by N4 SSM1/M2 -- not retested here)

Three configs, same instrumentation:
  A. UNPINNED steady state, n=N_MAIN -- reproduce the tail with per-call
     signals attached.
  B. PINNED to one core (os.sched_setaffinity) -- isolates whether the tail
     is migration-driven (hypothesis 4).
  C. NO-OP control -- identical loop/instrumentation shape, timing a
     do-nothing closure instead of score() -- isolates harness/scheduler
     noise (hypothesis 5) from anything intrinsic to score().

Instrumentation is OUTSIDE the timed window (t0/t1 wrap only the call being
measured), so latency_ns is not contaminated by the rusage/cpu reads
themselves -- those reads happen between iterations and describe what
happened during roughly that call's execution.

Run:
    /path/to/.venv/bin/python tail_cause.py            # full run
    /path/to/.venv/bin/python tail_cause.py --quick     # smoke test
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N1_PATH = HERE.parent / "single-record-overhead" / "n1_overhead.py"
T0 = time.perf_counter()


def elapsed() -> float:
    return time.perf_counter() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.2f}s] {msg}", flush=True)


def jsonl_append(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()


log(f"importing n1_overhead from {N1_PATH} (this pays the one-time kernel compile)")
_spec = importlib.util.spec_from_file_location("n1_overhead", str(N1_PATH))
n1 = importlib.util.module_from_spec(_spec)
sys.modules["n1_overhead"] = n1
_spec.loader.exec_module(n1)  # runs n1's module-level compile + sanity call
log(f"n1_overhead imported ok; rss={n1.rss_mb():.1f} MB")

RESULTS_JSONL = HERE / "results.jsonl"
if RESULTS_JSONL.exists():
    RESULTS_JSONL.unlink()


# ---------------------------------------------------------------------------
# CPU-core reader: /proc/self/stat field 39 ("processor"), held-open fd.
# ---------------------------------------------------------------------------
class CpuReader:
    def __init__(self):
        self._fd = os.open("/proc/self/stat", os.O_RDONLY)

    def current(self) -> int:
        os.lseek(self._fd, 0, 0)
        data = os.read(self._fd, 512)
        after = data.decode().rsplit(")", 1)[1].split()
        return int(after[39 - 3])

    def close(self):
        os.close(self._fd)


# ---------------------------------------------------------------------------
# Instrumented measurement loop
# ---------------------------------------------------------------------------
def measure(fn, n: int, warmup: int, label: str) -> dict:
    cpu = CpuReader()
    RUSAGE = resource.RUSAGE_SELF
    getrusage = resource.getrusage
    getblocks = sys.getallocatedblocks
    get_count = gc.get_count
    pcn = time.perf_counter_ns

    for _ in range(warmup):
        fn()

    lat = np.empty(n, dtype=np.int64)
    minflt = np.empty(n, dtype=np.int64)
    majflt = np.empty(n, dtype=np.int64)
    nvcsw = np.empty(n, dtype=np.int64)
    nivcsw = np.empty(n, dtype=np.int64)
    alloc_d = np.empty(n, dtype=np.int64)
    gc0_d = np.empty(n, dtype=np.int64)
    cpu_before = np.empty(n, dtype=np.int64)
    cpu_after = np.empty(n, dtype=np.int64)

    prev_cpu_after = cpu.current()
    for i in range(n):
        r0 = getrusage(RUSAGE)
        b0 = getblocks()
        g0 = get_count()[0]
        c_before = cpu.current()

        t0 = pcn()
        fn()
        t1 = pcn()

        c_after = cpu.current()
        r1 = getrusage(RUSAGE)
        b1 = getblocks()
        g1 = get_count()[0]

        lat[i] = t1 - t0
        minflt[i] = r1.ru_minflt - r0.ru_minflt
        majflt[i] = r1.ru_majflt - r0.ru_majflt
        nvcsw[i] = r1.ru_nvcsw - r0.ru_nvcsw
        nivcsw[i] = r1.ru_nivcsw - r0.ru_nivcsw
        alloc_d[i] = b1 - b0
        gc0_d[i] = g1 - g0
        cpu_before[i] = c_before
        cpu_after[i] = c_after

    cpu.close()

    out = {
        "label": label, "n": n,
        "lat_ns": lat, "minflt": minflt, "majflt": majflt,
        "nvcsw": nvcsw, "nivcsw": nivcsw, "alloc_d": alloc_d, "gc0_d": gc0_d,
        "cpu_before": cpu_before, "cpu_after": cpu_after,
    }
    return out


def pctiles_ns(ns: np.ndarray) -> dict:
    return {
        "p50_us": float(np.percentile(ns, 50)) / 1e3,
        "p95_us": float(np.percentile(ns, 95)) / 1e3,
        "p99_us": float(np.percentile(ns, 99)) / 1e3,
        "p99_9_us": float(np.percentile(ns, 99.9)) / 1e3,
        "max_us": float(ns.max()) / 1e3,
        "n": int(len(ns)),
    }


BUDGET_MS = 20.0


def pct_of_budget(us: float) -> float:
    return 100.0 * (us / 1e3) / BUDGET_MS


def migration_flags(cpu_before: np.ndarray, cpu_after: np.ndarray) -> np.ndarray:
    """1 if this call's own cpu_before != cpu_after (migrated mid-call) OR
    this call started on a different core than the previous call ended on
    (migrated between calls)."""
    mid_call = (cpu_before != cpu_after).astype(np.int64)
    between_calls = np.zeros_like(mid_call)
    between_calls[1:] = (cpu_before[1:] != cpu_after[:-1]).astype(np.int64)
    return np.maximum(mid_call, between_calls)


def correlate(result: dict) -> dict:
    lat = result["lat_ns"].astype(np.float64)
    signals = {
        "minflt": result["minflt"],
        "majflt": result["majflt"],
        "nvcsw": result["nvcsw"],
        "nivcsw": result["nivcsw"],
        "alloc_d": result["alloc_d"],
        "gc0_d": result["gc0_d"],
    }
    mig = migration_flags(result["cpu_before"], result["cpu_after"])
    signals["migration"] = mig

    p99 = float(np.percentile(lat, 99))
    p999 = float(np.percentile(lat, 99.9))
    slow99_mask = lat >= p99
    slow999_mask = lat >= p999
    overall_n = len(lat)

    out = {"label": result["label"], "n": overall_n,
           "p99_ns": p99, "p999_ns": p999,
           "n_slow99": int(slow99_mask.sum()), "n_slow999": int(slow999_mask.sum())}

    corr = {}
    rates = {}
    for name, sig in signals.items():
        sig_f = sig.astype(np.float64)
        if sig_f.std() > 0 and lat.std() > 0:
            r = float(np.corrcoef(lat, sig_f)[0, 1])
        else:
            r = 0.0
        corr[name] = r

        overall_rate = float((sig > 0).mean())
        slow99_rate = float((sig[slow99_mask] > 0).mean()) if slow99_mask.sum() else 0.0
        slow999_rate = float((sig[slow999_mask] > 0).mean()) if slow999_mask.sum() else 0.0
        rr99 = (slow99_rate / overall_rate) if overall_rate > 0 else (
            float("inf") if slow99_rate > 0 else 1.0)
        rates[name] = {
            "overall_rate_nonzero": overall_rate,
            "slow99_rate_nonzero": slow99_rate,
            "slow999_rate_nonzero": slow999_rate,
            "relative_risk_99": rr99,
            "overall_mean": float(sig_f.mean()),
            "slow99_mean": float(sig_f[slow99_mask].mean()) if slow99_mask.sum() else 0.0,
        }

    out["pearson_r_vs_latency"] = corr
    out["nonzero_rate_slow_vs_overall"] = rates
    return out


def run_config(fn, n: int, warmup: int, label: str, pin_core: int | None = None) -> dict:
    if pin_core is not None:
        os.sched_setaffinity(0, {pin_core})
        log(f"pinned to core {pin_core}; affinity now {os.sched_getaffinity(0)}")
    else:
        # make sure we're unpinned (full mask) in case a previous config pinned us
        try:
            all_cpus = set(range(os.cpu_count()))
            os.sched_setaffinity(0, all_cpus)
        except Exception as e:
            log(f"could not reset affinity: {e}")

    log(f"=== {label}: n={n} warmup={warmup} ===")
    result = measure(fn, n, warmup, label)
    pt = pctiles_ns(result["lat_ns"])
    log(f"  p50={pt['p50_us']:.1f}us p99={pt['p99_us']:.1f}us "
        f"p99.9={pt['p99_9_us']:.1f}us max={pt['max_us']:.1f}us "
        f"({pct_of_budget(pt['max_us']):.2f}% of {BUDGET_MS}ms budget)")
    corr = correlate(result)
    row = {"measurement": "config", "label": label, "pctiles": pt, "correlation": corr}
    jsonl_append(RESULTS_JSONL, row)

    # persist raw per-call arrays (small: n * 9 int64 columns) for offline re-analysis
    np.savez(HERE / f"raw_{label}.npz", **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    return {"pctiles": pt, "correlation": corr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-main", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--pin-core", type=int, default=2)
    args = ap.parse_args()

    n_main = 500 if args.quick else args.n_main
    warmup = 20 if args.quick else args.warmup

    def call_score():
        n1.score(n1.REQUEST_DICT, n1.PARAMS_RAW)

    def call_noop():
        pass

    summary = {}
    summary["A_unpinned"] = run_config(call_score, n_main, warmup, "A_unpinned", pin_core=None)
    summary["B_pinned"] = run_config(call_score, n_main, warmup, "B_pinned", pin_core=args.pin_core)
    # reset to unpinned for the no-op control (harness-noise baseline should
    # reflect the same scheduling conditions as config A)
    summary["C_noop_unpinned"] = run_config(call_noop, n_main, warmup, "C_noop_unpinned", pin_core=None)

    (HERE / "results_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"DONE -- rss now {n1.rss_mb():.1f} MB")


if __name__ == "__main__":
    main()
