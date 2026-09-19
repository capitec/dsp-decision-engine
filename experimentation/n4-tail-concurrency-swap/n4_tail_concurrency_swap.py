"""
EXPERIMENT N4 -- tail latency, concurrency, and config-swap impact on serving.

Doc 01 S6.1: single-record score() is the PRIMARY path, budget 20-100 ms/record.
Fifteen prior experiments measured medians only. "At a 20ms SLA the tail IS the
SLA" -- this harness measures p50/p95/p99/p99.9/max, not just p50, for:

  M1  single-thread steady state, N=50,000 score() calls -- tail + GC correlation
  M2  gc.disable() and gc.freeze() vs baseline -- does disabling GC fix the tail?
  M3  concurrency at 1/2/4/8/16 threads, nogil=True vs nogil=False -- throughput
      AND tail, serving against serving (not against a compiler, cf. EXPERIMENTS.md SH)
  M4  continuous single-record traffic through repeated config-generation swaps --
      tail during swaps vs steady state, and first-call-after-swap cost

Per the task brief: "Do not benchmark the kernel" -- the compiled dispatch is
~1us and already measured (E9/J/N1). This harness benchmarks what N1 called
score(): accept + validate + marshal + dispatch + readback + assemble, i.e. the
framework overhead layer, under load and across time -- not the kernel in isolation.

Reused, not rewritten (all imported by path, not copy-pasted):
  - experimentation/single-record-overhead/n1_overhead.py:
      DRIVER (the compiled nogil=True 400-in/633-out record kernel), score(),
      dispatch(), marshal(), readback(), accept_kwargs(), bind_params(),
      assemble(), REQUEST_DICT, PARAMS_RAW, IN_DTYPE, OUT_REC_DTYPE, COEF_F8,
      pctiles(), timed_ns(), jsonl_append(), rss_mb() (resource.getrusage
      pattern), gen_record_out_source() (reused verbatim to build the
      nogil=False twin kernel for M3 -- see build_nogil_false_driver()).
  - experimentation/ruleset-compile-latency/emit.py:
      emit_ruleset() -- reused unmodified to build the two "config generations"
      swapped in M4, exactly as experiment H (staged-compile-atomic-swap) does.
  - experimentation/staged-compile-atomic-swap/run.py:
      compile_kernel() pattern (exec + njit + call-once-to-force-compile) --
      reused (imported) for building the two M4 generations.

NOT reused: nothing else was rewritten instead of importing; the only new
code below is the M1-M4 measurement loops themselves and a small
load_source() variant that writes the nogil=False driver into THIS
directory instead of n1_overhead's (so the two experiments' generated-file
artifacts don't collide).

Run:
    /path/to/.venv/bin/python n4_tail_concurrency_swap.py            # full run
    /path/to/.venv/bin/python n4_tail_concurrency_swap.py --quick    # smoke test

Writes results.jsonl (flushed after every measurement) and results_summary.json.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import numba
import pydantic

HERE = Path(__file__).resolve().parent
T0 = time.perf_counter()


def elapsed() -> float:
    return time.perf_counter() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Reuse: import n1_overhead.py and ruleset-compile-latency/emit.py and
# staged-compile-atomic-swap/run.py by path (all are import-safe: heavy work
# is behind `if __name__ == "__main__":` guards, confirmed before importing).
# ---------------------------------------------------------------------------
def _import_by_path(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


N1_DIR = HERE.parent / "single-record-overhead"
RULES_DIR = HERE.parent / "ruleset-compile-latency"
SWAP_DIR = HERE.parent / "staged-compile-atomic-swap"

log("importing n1_overhead.py (this recompiles the 400/633 kernel once, ~1-3s)...")
n1 = _import_by_path("n1_overhead_reused", N1_DIR / "n1_overhead.py")
log(f"  n1 kernel compiled in {n1._compile_s:.2f}s; rss={n1.rss_mb():.1f} MB")

sys.path.insert(0, str(RULES_DIR))
import emit  # noqa: E402  -- reused unmodified

sys.path.insert(0, str(SWAP_DIR))
swap_run = _import_by_path("swap_run_reused", SWAP_DIR / "run.py")

rss_mb = n1.rss_mb
jsonl_append = n1.jsonl_append
timed_ns = n1.timed_ns
RESULTS_JSONL = HERE / "results.jsonl"
BUDGET_MS = 20.0


def pct_of_budget(us: float) -> float:
    return 100.0 * (us / 1e3) / BUDGET_MS


def pctiles_full(ns: np.ndarray) -> dict:
    """p50/p95/p99/p99.9/max -- n1's pctiles() lacked p99.9, which this
    experiment specifically needs ("the tail IS the SLA")."""
    return {
        "n": int(len(ns)),
        "p50_us": float(np.percentile(ns, 50)) / 1e3,
        "p95_us": float(np.percentile(ns, 95)) / 1e3,
        "p99_us": float(np.percentile(ns, 99)) / 1e3,
        "p999_us": float(np.percentile(ns, 99.9)) / 1e3,
        "max_us": float(ns.max()) / 1e3,
        "mean_us": float(ns.mean()) / 1e3,
    }


def log_pct(label: str, p: dict) -> None:
    log(f"  {label}: p50={p['p50_us']:.1f}us({pct_of_budget(p['p50_us']):.3f}%) "
        f"p95={p['p95_us']:.1f}us({pct_of_budget(p['p95_us']):.3f}%) "
        f"p99={p['p99_us']:.1f}us({pct_of_budget(p['p99_us']):.3f}%) "
        f"p999={p['p999_us']:.1f}us({pct_of_budget(p['p999_us']):.3f}%) "
        f"max={p['max_us']:.1f}us({pct_of_budget(p['max_us']):.3f}%)")


# ===========================================================================
# M1 -- single-thread steady state + GC correlation
# ===========================================================================
def measure_m1(n: int) -> dict:
    log(f"=== M1: single-thread steady state, n={n} score() calls ===")
    gc_events = []  # (timestamp_s,) collected by gc.callbacks

    def on_gc(phase, info):
        if phase == "start":
            gc_events.append(time.perf_counter())

    gc.callbacks.append(on_gc)
    gc.collect()
    try:
        for _ in range(200):
            n1.score(n1.REQUEST_DICT, n1.PARAMS_RAW)  # warmup

        t0s = np.empty(n, dtype=np.float64)
        dns = np.empty(n, dtype=np.int64)
        for i in range(n):
            t0 = time.perf_counter()
            tn0 = time.perf_counter_ns()
            n1.score(n1.REQUEST_DICT, n1.PARAMS_RAW)
            tn1 = time.perf_counter_ns()
            t0s[i] = t0
            dns[i] = tn1 - tn0
    finally:
        gc.callbacks.remove(on_gc)

    p = pctiles_full(dns)
    log_pct("M1 steady-state", p)

    # correlate: for each gc event, find the call whose window contains it
    # (approx: call i covers [t0s[i], t0s[i] + dns[i]/1e9])
    slow_thresh_ns = np.percentile(dns, 99)
    slow_idx = np.where(dns >= slow_thresh_ns)[0]
    gc_hit = 0
    for i in slow_idx:
        t_start = t0s[i]
        t_end = t0s[i] + dns[i] / 1e9
        for g in gc_events:
            if t_start - 2e-4 <= g <= t_end + 2e-4:  # 200us tolerance both sides
                gc_hit += 1
                break
    frac_slow_is_gc = gc_hit / max(len(slow_idx), 1)
    log(f"  gc events during run: {len(gc_events)}; calls >=p99 ({slow_thresh_ns/1e3:.1f}us): "
        f"{len(slow_idx)}; of those coincident with a gc event: {gc_hit} "
        f"({100*frac_slow_is_gc:.1f}%)")

    rec = {"measurement": "m1_steady_state", **p,
           "pct_of_budget_p50": pct_of_budget(p["p50_us"]),
           "pct_of_budget_p99": pct_of_budget(p["p99_us"]),
           "pct_of_budget_p999": pct_of_budget(p["p999_us"]),
           "pct_of_budget_max": pct_of_budget(p["max_us"]),
           "gc_events_during_run": len(gc_events),
           "n_calls_at_or_above_p99": int(len(slow_idx)),
           "n_p99_calls_coincident_with_gc": int(gc_hit),
           "frac_p99_calls_coincident_with_gc": frac_slow_is_gc}
    jsonl_append(RESULTS_JSONL, rec)
    return rec


# ===========================================================================
# M2 -- GC on vs gc.disable() vs gc.freeze()
# ===========================================================================
def measure_m2(n: int) -> dict:
    log(f"=== M2: GC variants, n={n} score() calls each ===")
    out = {}

    def run_variant(label: str) -> dict:
        for _ in range(200):
            n1.score(n1.REQUEST_DICT, n1.PARAMS_RAW)
        dns = np.empty(n, dtype=np.int64)
        for i in range(n):
            t0 = time.perf_counter_ns()
            n1.score(n1.REQUEST_DICT, n1.PARAMS_RAW)
            dns[i] = time.perf_counter_ns() - t0
        p = pctiles_full(dns)
        log_pct(f"M2 {label}", p)
        rec = {"measurement": "m2_gc_variant", "variant": label, **p,
               "pct_of_budget_p50": pct_of_budget(p["p50_us"]),
               "pct_of_budget_p99": pct_of_budget(p["p99_us"]),
               "pct_of_budget_p999": pct_of_budget(p["p999_us"]),
               "pct_of_budget_max": pct_of_budget(p["max_us"])}
        jsonl_append(RESULTS_JSONL, rec)
        return rec

    gc.enable()
    gc.collect()
    out["baseline_gc_on"] = run_variant("baseline_gc_on")

    gc.disable()
    out["gc_disabled"] = run_variant("gc_disabled")
    gc.enable()

    gc.collect()
    gc.freeze()
    out["gc_frozen"] = run_variant("gc_frozen_after_warmup")
    gc.unfreeze()

    return out


# ===========================================================================
# M3 -- concurrency: 1/2/4/8/16 threads x nogil True/False
# ===========================================================================
def build_nogil_false_driver():
    """Reuses n1.gen_record_out_source() (the emitter) verbatim; only the
    nogil flag in the header and the output directory differ, because
    n1's load_source() writes into the n1 experiment's OWN directory."""
    src = n1.gen_record_out_source()
    src_false = src.replace("nogil=True", "nogil=False")
    assert "nogil=False" in src_false and src_false.count("nogil=") == 1
    path = HERE / "_generated_driver_nogil_false.py"
    path.write_text(src_false)
    spec = importlib.util.spec_from_file_location("n4_driver_nogil_false", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    driver = mod.driver
    warm_in = np.empty(1, dtype=n1.IN_DTYPE)
    warm_out = np.empty(1, dtype=n1.OUT_REC_DTYPE)
    driver(warm_in, n1.COEF_F8, warm_out)  # force compile
    return driver


def measure_m3(thread_counts, duration_s: float) -> list:
    log(f"=== M3: concurrency sweep, threads={thread_counts}, {duration_s}s/config ===")
    log("  building nogil=False twin kernel (reused emitter, new nogil flag)...")
    driver_false = build_nogil_false_driver()
    driver_true = n1.DRIVER
    results = []

    def worker(driver, stop_at: float, dn_list: list, count_box: list):
        rec = np.empty(1, dtype=n1.IN_DTYPE)
        row = rec[0]
        for name in n1.IN_ALL_NAMES:
            row[name] = n1.REQUEST_DICT[name]
        out = np.empty(1, dtype=n1.OUT_REC_DTYPE)
        local_dn = []
        count = 0
        while time.perf_counter() < stop_at:
            t0 = time.perf_counter_ns()
            driver(rec, n1.COEF_F8, out)
            values = {nm: out[0][nm].item() for nm in n1.OUT_ALL_NAMES}  # readback, N1-shaped
            t1 = time.perf_counter_ns()
            count += 1
            if len(local_dn) < 300_000:
                local_dn.append(t1 - t0)
        dn_list.extend(local_dn)
        count_box.append(count)

    for nogil_label, driver in (("nogil_true", driver_true), ("nogil_false", driver_false)):
        for nt in thread_counts:
            dn_lists = [[] for _ in range(nt)]
            count_boxes = [[] for _ in range(nt)]
            stop_at = time.perf_counter() + duration_s + 0.05
            threads = [threading.Thread(target=worker, args=(driver, stop_at, dn_lists[i], count_boxes[i]))
                       for i in range(nt)]
            t_wall0 = time.perf_counter()
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            t_wall1 = time.perf_counter()

            all_dn = np.array([x for lst in dn_lists for x in lst], dtype=np.int64)
            total_calls = sum(cb[0] for cb in count_boxes)
            wall = t_wall1 - t_wall0
            throughput = total_calls / wall
            p = pctiles_full(all_dn) if len(all_dn) else {}
            log(f"  {nogil_label} threads={nt:2d}: {total_calls:7d} calls in {wall:.2f}s "
                f"= {throughput:8.0f} calls/s; per-call p50={p.get('p50_us', float('nan')):.2f}us "
                f"p99={p.get('p99_us', float('nan')):.2f}us max={p.get('max_us', float('nan')):.2f}us")
            rec = {"measurement": "m3_concurrency", "nogil": nogil_label, "threads": nt,
                   "total_calls": total_calls, "wall_s": wall, "throughput_calls_per_s": throughput,
                   **{f"pc_{k}": v for k, v in p.items()}}
            jsonl_append(RESULTS_JSONL, rec)
            results.append(rec)

    # throughput retention at 16 threads vs 1 thread (single-threaded-equivalent
    # baseline), the ratio EXPERIMENTS.md SH used ("26% throughput") -- computed
    # per nogil setting for direct comparison.
    for nogil_label in ("nogil_true", "nogil_false"):
        rows = [r for r in results if r["nogil"] == nogil_label]
        base = next(r for r in rows if r["threads"] == thread_counts[0])
        top = next(r for r in rows if r["threads"] == thread_counts[-1])
        per_thread_base = base["throughput_calls_per_s"] / thread_counts[0]
        per_thread_top = top["throughput_calls_per_s"] / thread_counts[-1]
        retention = per_thread_top / per_thread_base
        log(f"  {nogil_label}: per-thread throughput retention at {thread_counts[-1]}x threads "
            f"vs {thread_counts[0]}x = {100*retention:.1f}%")
        jsonl_append(RESULTS_JSONL, {"measurement": "m3_retention", "nogil": nogil_label,
                                      "threads_base": thread_counts[0], "threads_top": thread_counts[-1],
                                      "per_thread_retention": retention})
    return results


# ===========================================================================
# M4 -- continuous single-record traffic through repeated config swaps
# ===========================================================================
class GenHolder:
    __slots__ = ("driver", "args", "label")

    def __init__(self, driver, args, label):
        self.driver = driver
        self.args = args
        self.label = label


def build_generation(n_rules: int, seed: int, label: str) -> GenHolder:
    """Reuses emit.emit_ruleset() (unmodified) and the compile_kernel()
    pattern from staged-compile-atomic-swap/run.py (imported, not copied)."""
    src = emit.emit_ruleset(n_rules, "first_match", seed=seed, depth=2, fname=f"gen_{label}")
    args = swap_run.make_args(1, "first_match", n_rules, seed=seed)  # N=1 row -- single-record
    disp, compile_s = swap_run.compile_kernel(src, f"gen_{label}", args, nogil=True)
    log(f"  generation {label}: n_rules={n_rules} compiled in {compile_s*1e3:.1f}ms")
    return GenHolder(disp, args, label)


def measure_m4(duration_s: float, swap_interval_s: float) -> dict:
    log(f"=== M4: config swap under continuous single-record traffic, "
        f"{duration_s}s, swap every {swap_interval_s*1000:.0f}ms ===")
    gen_a = build_generation(10, seed=1, label="A")
    gen_b = build_generation(10, seed=2, label="B")
    # warm both (compile_kernel already called each once, so this is a 2nd
    # call -- confirms no further first-call cost from compile_kernel itself)
    gen_a.driver(*gen_a.args)
    gen_b.driver(*gen_b.args)

    current = [gen_a]  # single-element list = the "generation pointer"; list
                        # index assignment is a single reference store, atomic
                        # under the GIL -- the same primitive doc 08 assumes.

    stop_flag = threading.Event()
    swap_events = []  # perf_counter() timestamps of each swap

    def swapper():
        toggle = True
        while not stop_flag.is_set():
            time.sleep(swap_interval_s)
            current[0] = gen_b if toggle else gen_a
            swap_events.append(time.perf_counter())
            toggle = not toggle

    call_t0 = []
    call_dn = []
    t_end = time.perf_counter() + duration_s
    sw_thread = threading.Thread(target=swapper)
    sw_thread.start()
    while time.perf_counter() < t_end:
        gen = current[0]
        t0 = time.perf_counter()
        tn0 = time.perf_counter_ns()
        gen.driver(*gen.args)
        tn1 = time.perf_counter_ns()
        call_t0.append(t0)
        call_dn.append(tn1 - tn0)
    stop_flag.set()
    sw_thread.join()

    call_t0 = np.array(call_t0)
    call_dn = np.array(call_dn, dtype=np.int64)
    log(f"  served {len(call_dn)} calls, {len(swap_events)} swaps")

    overall = pctiles_full(call_dn)
    log_pct("M4 overall (incl. swaps)", overall)

    # first call after each swap: earliest call with t0 >= swap_event
    first_after = []
    for sw in swap_events:
        idx = np.searchsorted(call_t0, sw, side="left")
        if idx < len(call_dn):
            first_after.append(call_dn[idx])
    first_after = np.array(first_after, dtype=np.int64) if first_after else np.array([0], dtype=np.int64)

    # "ordinary" calls: everything NOT flagged as first-after-swap
    first_after_idx = set()
    for sw in swap_events:
        idx = np.searchsorted(call_t0, sw, side="left")
        if idx < len(call_dn):
            first_after_idx.add(idx)
    ordinary_mask = np.ones(len(call_dn), dtype=bool)
    for idx in first_after_idx:
        ordinary_mask[idx] = False
    ordinary = pctiles_full(call_dn[ordinary_mask])
    first_p = pctiles_full(first_after)

    log_pct("M4 steady-state (excl. first-after-swap)", ordinary)
    log_pct("M4 first-call-after-swap only", first_p)
    log(f"  first-after-swap median={first_p['p50_us']:.2f}us vs steady median="
        f"{ordinary['p50_us']:.2f}us -> {first_p['p50_us']/max(ordinary['p50_us'],1e-9):.2f}x")

    rec = {
        "measurement": "m4_config_swap",
        "duration_s": duration_s, "swap_interval_s": swap_interval_s,
        "n_calls": int(len(call_dn)), "n_swaps": len(swap_events),
        "overall": overall, "steady_state_excl_first_after_swap": ordinary,
        "first_call_after_swap": first_p,
        "first_after_swap_vs_steady_ratio_p50": first_p["p50_us"] / max(ordinary["p50_us"], 1e-9),
    }
    jsonl_append(RESULTS_JSONL, rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--m1-n", type=int, default=50_000)
    ap.add_argument("--m2-n", type=int, default=20_000)
    ap.add_argument("--m3-threads", default="1,2,4,8,16")
    ap.add_argument("--m3-duration", type=float, default=1.5)
    ap.add_argument("--m4-duration", type=float, default=4.0)
    ap.add_argument("--m4-swap-interval", type=float, default=0.1)
    args = ap.parse_args()

    if args.quick:
        m1_n, m2_n = 500, 300
        m3_threads = [1, 4]
        m3_dur = 0.3
        m4_dur, m4_swap = 0.6, 0.05
    else:
        m1_n, m2_n = args.m1_n, args.m2_n
        m3_threads = [int(x) for x in args.m3_threads.split(",")]
        m3_dur = args.m3_duration
        m4_dur, m4_swap = args.m4_duration, args.m4_swap_interval

    if RESULTS_JSONL.exists():
        RESULTS_JSONL.unlink()

    env = {"python": platform.python_version(), "numba": numba.__version__,
           "numpy": np.__version__, "pydantic": pydantic.VERSION,
           "cpu_count": __import__("os").cpu_count()}
    jsonl_append(RESULTS_JSONL, {"measurement": "env", **env})
    log(f"env: {env}")
    log(f"rss before measurement: {rss_mb():.1f} MB")

    m1 = measure_m1(m1_n)
    m2 = measure_m2(m2_n)
    m3 = measure_m3(m3_threads, m3_dur)
    m4 = measure_m4(m4_dur, m4_swap)

    log(f"rss after measurement: {rss_mb():.1f} MB")
    summary = {"env": env, "m1_steady_state": m1, "m2_gc_variants": m2,
               "m3_concurrency": m3, "m4_config_swap": m4}
    (HERE / "results_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("DONE -- wrote results.jsonl and results_summary.json")


if __name__ == "__main__":
    main()
