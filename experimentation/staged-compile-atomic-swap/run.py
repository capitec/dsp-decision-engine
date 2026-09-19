"""Experiment H — staged compilation and atomic swap (doc 08 §4).

Measures whether doc 08 §4's lifecycle
    ACTIVE --stage(structure)--> COMPILING (background) --> STAGED --activate()--> ACTIVE
    "the previous generation keeps serving throughout"
    "apply and score read the generation pointer exactly once per invocation"
is implementable with numba in-process.

Phases
  interference : background numba compile vs. a serving kernel in the main thread,
                 for nogil=True and nogil=False serving kernels.
  lockstep     : does a *lazy* main-thread compile block behind a background compile?
                 do two background compiles run in parallel?
  swap         : generation-pointer swap under concurrent load; straddle detection;
                 rollback compile count; swap latency.
  memory       : RSS with 1, 2, 3 compiled generations resident (own subprocess).

Run:  .venv/bin/python run.py            # all phases, writes results.json
      .venv/bin/python run.py --phase swap
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Rule-set emitter is shared with experiment G (ruleset-compile-latency); importing
# it rather than duplicating keeps the emitted shape identical across experiments.
sys.path.insert(0, str(HERE.parent / "ruleset-compile-latency"))

import emit  # noqa: E402
import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

try:
    from numba.core import event as nb_event
except Exception:  # pragma: no cover
    nb_event = None

ROWS = 100_000


# --------------------------------------------------------------------------- utils
def rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def make_args(rows: int, mode: str, n_rules: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    fs = [np.ascontiguousarray(rng.random(rows)) for _ in range(emit.N_FLOAT)]
    cs = [np.ascontiguousarray(rng.integers(0, 10, rows)) for _ in range(emit.N_CODE)]
    bs = [np.ascontiguousarray(rng.random(rows) > 0.5) for _ in range(emit.N_BOOL)]
    out = np.zeros(rows, np.int64) if mode == "first_match" else np.zeros((rows, n_rules), np.int64)
    return tuple(fs + cs + bs + [out])


def build_src(n_rules, mode, seed, fname):
    return emit.emit_ruleset(n_rules, mode, seed=seed, depth=2, fname=fname)


def compile_kernel(src, fname, args, nogil=True):
    """exec + njit + force compilation by calling once. Returns (disp, seconds)."""
    ns: dict = {}
    exec(compile(src, f"<{fname}>", "exec"), ns)
    disp = njit(nogil=nogil, cache=False)(ns[fname])
    t0 = time.perf_counter()
    disp(*args)
    return disp, time.perf_counter() - t0


def count_compiles(fn):
    """Run fn(); return (result, n_numba_compile_events) using numba's event API."""
    if nb_event is None:
        return fn(), None
    try:
        with nb_event.install_recorder("numba:compile") as rec:
            r = fn()
        return r, len(rec.buffer)
    except Exception:
        return fn(), None


# --------------------------------------------------------- phase: interference / GIL
def serve_until(kernel, args, stop: threading.Event, max_s: float):
    times = []
    t_end = time.perf_counter() + max_s
    while not stop.is_set() and time.perf_counter() < t_end:
        t0 = time.perf_counter()
        kernel(*args)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def phase_interference(reps=2, baseline_s=2.0, bg_rules=30, serve_rules=10):
    """Q1/Q2/Q3: background compile while serving; nogil True vs False."""
    res = {"bg_rules": bg_rules, "serve_rules": serve_rules, "reps": reps, "modes": {}}

    serve_args = make_args(ROWS, "first_match", serve_rules)
    serve_src = build_src(serve_rules, "first_match", seed=101, fname="serve")
    bg_args = make_args(ROWS, "first_match", bg_rules)
    # identical source for every background compile, so each repeat is the same work
    bg_src = build_src(bg_rules, "first_match", seed=202, fname="bg")
    res["bg_emitted_lines"] = emit.count_lines(bg_src)
    res["serve_emitted_lines"] = emit.count_lines(serve_src)

    # compile-alone reference (no serving thread, nothing else running)
    alone = []
    ref_disp = None
    for _ in range(reps):
        ref_disp, s = compile_kernel(bg_src, "bg", bg_args, nogil=True)
        alone.append(s)
    ref_out = np.array(bg_args[-1])  # output of the single-threaded compile of bg_src
    res["compile_alone_s"] = alone
    res["compile_alone_median_s"] = med(alone)

    for nogil in (True, False):
        k, csec = compile_kernel(serve_src, "serve", serve_args, nogil=nogil)
        for _ in range(5):
            k(*serve_args)  # warm

        base = serve_until(k, serve_args, threading.Event(), baseline_s)
        mode = {
            "serving_compile_s": csec,
            "baseline_calls": len(base),
            "baseline_median_ms": med(base),
            "baseline_p95_ms": pct(base, 95),
            "baseline_calls_per_s": len(base) / (sum(base) / 1e3),
            "runs": [],
        }

        for rep in range(reps):
            stop = threading.Event()
            box: dict = {}

            def worker():
                t0 = time.perf_counter()
                try:
                    bg_out = np.zeros_like(ref_out)
                    d, _ = compile_kernel(bg_src, "bg", bg_args[:-1] + (bg_out,), nogil=True)
                    box["ok"] = True
                    box["output_matches_singlethreaded_compile"] = bool(np.array_equal(bg_out, ref_out))
                except BaseException as exc:  # noqa: BLE001 - we want to report any failure
                    box["ok"] = False
                    box["error"] = f"{type(exc).__name__}: {exc}"
                box["compile_s"] = time.perf_counter() - t0
                stop.set()

            th = threading.Thread(target=worker, name="stage-compiler", daemon=True)
            t0 = time.perf_counter()
            th.start()
            during = serve_until(k, serve_args, stop, 60.0)
            th.join(timeout=60)
            wall = time.perf_counter() - t0
            mode["runs"].append(
                {
                    "compile_ok": box.get("ok"),
                    "compile_error": box.get("error"),
                    "compile_s_during_serving": box.get("compile_s"),
                    "bg_output_matches_singlethreaded": box.get("output_matches_singlethreaded_compile"),
                    "wall_s": wall,
                    "calls": len(during),
                    "median_ms": med(during),
                    "p95_ms": pct(during, 95),
                    "max_ms": max(during) if during else None,
                    "calls_per_s": len(during) / wall,
                }
            )
        b_cps = mode["baseline_calls"] / baseline_s
        d_cps = med([r["calls_per_s"] for r in mode["runs"]])
        mode["baseline_calls_per_s_wall"] = b_cps
        mode["during_calls_per_s_median"] = d_cps
        mode["throughput_retained"] = d_cps / b_cps
        mode["latency_inflation_median"] = med([r["median_ms"] for r in mode["runs"]]) / mode["baseline_median_ms"]
        mode["compile_slowdown_vs_alone"] = (
            med([r["compile_s_during_serving"] for r in mode["runs"]]) / res["compile_alone_median_s"]
        )
        res["modes"]["nogil=%s" % nogil] = mode
    return res


def phase_gilprobe(bg_rules=30, serve_rules=10, baseline_s=1.0):
    """Q3 follow-up: is the nogil serving penalty a GIL hand-off cost?

    If it is, it is a FIXED cost per invocation (~one switch interval), so it must
    shrink with sys.setswitchinterval and amortise over a larger batch.
    """
    bg_src = build_src(bg_rules, "first_match", seed=202, fname="bg")
    serve_src = build_src(serve_rules, "first_match", seed=101, fname="serve")
    runs = []

    def one(rows, switch_s, label):
        args = make_args(rows, "first_match", serve_rules)
        bg_args = make_args(ROWS, "first_match", bg_rules)
        k, _ = compile_kernel(serve_src, "serve", args, nogil=True)
        for _ in range(5):
            k(*args)
        old = sys.getswitchinterval()
        sys.setswitchinterval(switch_s)
        try:
            base = serve_until(k, args, threading.Event(), baseline_s)
            stop = threading.Event()

            def bg():
                compile_kernel(bg_src, "bg", bg_args, nogil=True)
                stop.set()

            th = threading.Thread(target=bg, daemon=True)
            th.start()
            during = serve_until(k, args, stop, 60.0)
            th.join(60)
        finally:
            sys.setswitchinterval(old)
        return {
            "label": label, "rows": rows, "switchinterval_s": switch_s,
            "baseline_median_ms": med(base), "during_median_ms": med(during),
            "during_p95_ms": pct(during, 95), "during_max_ms": max(during) if during else None,
            "added_ms_per_call": med(during) - med(base),
            "latency_inflation": med(during) / med(base),
        }

    for sw in (0.005, 0.0005, 0.00005):
        runs.append(one(ROWS, sw, f"rows=100k switch={sw*1e3:g}ms"))
    runs.append(one(1_000_000, 0.005, "rows=1M switch=5ms"))
    return {"runs": runs}


# ------------------------------------------------------------------ phase: lockstep
def phase_lockstep(big_rules=30, small_rules=3, par_rules=10):
    """Q3b: does numba's global compiler lock serialise compiles?"""
    res = {}
    big_args = make_args(ROWS, "first_match", big_rules)
    big_src = build_src(big_rules, "first_match", seed=303, fname="big")
    small_args = make_args(ROWS, "first_match", small_rules)
    small_src = build_src(small_rules, "first_match", seed=404, fname="small")

    alone = [compile_kernel(small_src, "small", small_args)[1] for _ in range(2)]
    res["small_compile_alone_s"] = alone
    res["small_compile_alone_median_s"] = med(alone)

    blocked = []
    for _ in range(2):
        done = threading.Event()

        def bg():
            compile_kernel(big_src, "big", big_args)
            done.set()

        th = threading.Thread(target=bg, daemon=True)
        t_bg = time.perf_counter()
        th.start()
        time.sleep(0.5)  # let the background compile get into the compiler
        t0 = time.perf_counter()
        _, _ = compile_kernel(small_src, "small", small_args)
        dt = time.perf_counter() - t0
        th.join(60)
        blocked.append({"small_s": dt, "bg_total_s": time.perf_counter() - t_bg})
    res["small_compile_while_bg"] = blocked
    res["small_compile_while_bg_median_s"] = med([b["small_s"] for b in blocked])
    res["small_blocking_factor"] = res["small_compile_while_bg_median_s"] / res["small_compile_alone_median_s"]

    # two "stage" compiles at once vs. one after the other
    par_args = make_args(ROWS, "first_match", par_rules)
    srcs = [build_src(par_rules, "first_match", seed=500 + i, fname=f"par{i}") for i in range(2)]
    t0 = time.perf_counter()
    for i, s in enumerate(srcs):
        compile_kernel(s, f"par{i}", par_args)
    res["two_compiles_sequential_s"] = time.perf_counter() - t0

    ths = []
    t0 = time.perf_counter()
    for i, s in enumerate(srcs):
        th = threading.Thread(target=compile_kernel, args=(s, f"par{i}", par_args), daemon=True)
        th.start()
        ths.append(th)
    for th in ths:
        th.join(120)
    res["two_compiles_parallel_s"] = time.perf_counter() - t0
    res["parallel_speedup"] = res["two_compiles_sequential_s"] / res["two_compiles_parallel_s"]
    return res


# ---------------------------------------------------------------------- phase: swap
GEN_SRC = """
def gen_kernel_{tag}(x, out):
    n = out.shape[0]
    for i in range(n):
        out[i] = {tag} * 1000000 + (int(x[i] * 1000.0) % 997)
"""


class Generation:
    __slots__ = ("gen_id", "kernel")

    def __init__(self, gen_id, kernel):
        self.gen_id = gen_id
        self.kernel = kernel


class Runtime:
    """doc 08 §4: one active generation, explicit activate, free rollback."""

    def __init__(self, gen):
        self._gen = gen
        self._prev = None

    @property
    def generation(self):
        return self._gen  # one attribute read == the pointer read

    def activate(self, gen):
        self._prev, self._gen = self._gen, gen

    def rollback(self):
        if self._prev is None:
            raise RuntimeError("nothing to roll back to")
        self._gen, self._prev = self._prev, self._gen


def phase_swap(rows=200_000, workers=4, duration_s=3.0, swap_every_s=0.0002, chunks=10):
    res = {"rows_per_batch": rows, "workers": workers, "duration_s": duration_s,
           "swap_every_s": swap_every_s, "straddling_chunks": chunks}
    x = np.ascontiguousarray(np.random.default_rng(3).random(rows))

    gens = []
    compile_s = []
    for tag in (1, 2):
        ns: dict = {}
        exec(GEN_SRC.format(tag=tag), ns)
        disp = njit(nogil=True, cache=False)(ns[f"gen_kernel_{tag}"])
        o = np.zeros(rows, np.int64)
        t0 = time.perf_counter()
        disp(x, o)
        compile_s.append(time.perf_counter() - t0)
        gens.append(Generation(tag, disp))
    res["generation_compile_s"] = compile_s

    o = np.zeros(rows, np.int64)
    t0 = time.perf_counter()
    for _ in range(20):
        gens[0].kernel(x, o)
    res["batch_ms"] = (time.perf_counter() - t0) / 20 * 1e3

    def run(correct: bool):
        rt = Runtime(gens[0])
        stop = threading.Event()
        stats = {"batches": 0, "straddled": 0, "swaps": 0}
        lock = threading.Lock()

        def swapper():
            k = 0
            while not stop.is_set():
                k += 1
                rt.activate(gens[k % 2])
                time.sleep(swap_every_s)
            stats["swaps"] = k

        def worker():
            out = np.zeros(rows, np.int64)
            nb = strad = 0
            while not stop.is_set():
                if correct:
                    gen = rt.generation          # read ONCE per invocation
                    gen.kernel(x, out)
                else:
                    step = rows // chunks
                    for c in range(chunks):     # re-read the pointer per chunk
                        a, b = c * step, (c + 1) * step
                        rt.generation.kernel(x[a:b], out[a:b])
                nb += 1
                if len(np.unique(out // 1000000)) != 1:
                    strad += 1
            with lock:
                stats["batches"] += nb
                stats["straddled"] += strad

        sw = threading.Thread(target=swapper, daemon=True)
        ws = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
        sw.start()
        for w in ws:
            w.start()
        time.sleep(duration_s)
        stop.set()
        for w in ws:
            w.join(30)
        sw.join(30)
        return stats

    res["correct_read_once"] = run(True)
    res["straddling_control"] = run(False)
    res["control_has_power"] = res["straddling_control"]["straddled"] > 0

    # swap latency + rollback
    rt = Runtime(gens[0])
    lat = []
    for i in range(10000):
        t0 = time.perf_counter()
        rt.activate(gens[i % 2])
        lat.append((time.perf_counter() - t0) * 1e6)
    res["activate_latency_us_median"] = med(lat)
    res["activate_latency_us_p99"] = pct(lat, 99)

    rt = Runtime(gens[0])
    rt.activate(gens[1])
    out = np.zeros(rows, np.int64)

    def do_rollback():
        t0 = time.perf_counter()
        rt.rollback()
        dt = time.perf_counter() - t0
        for _ in range(5):
            rt.generation.kernel(x, out)
        return dt, int(out[0] // 1000000)

    (dt, tag_after), n_comp = count_compiles(do_rollback)
    res["rollback_us"] = dt * 1e6
    res["rollback_numba_compiles"] = n_comp
    res["rollback_serves_generation"] = tag_after
    res["rollback_correct"] = tag_after == gens[0].gen_id
    return res


# ------------------------------------------------------------------ phase: subproc
CHILD = """
import sys, time
sys.path.insert(0, {gen!r}); sys.path.insert(0, {here!r})
import numpy as np
from numba import njit
import run as h
import staged_gen_kernel as m
args = h.make_args(h.ROWS, "first_match", {rules})
d = njit(cache=True)(m.kern)
t0 = time.perf_counter(); d(*args); print("COMPILE_S", time.perf_counter() - t0)
print("STATS", d.stats)
print("CHECKSUM", int(args[-1].sum()))
"""


def phase_subproc(bg_rules=30, serve_rules=10, baseline_s=1.0):
    """The alternative to a compiler thread: compile in a child process, hand the
    result over through numba's file-backed cache. Costs the cache's fragility."""
    gen_dir = HERE / "_gen"
    gen_dir.mkdir(exist_ok=True)
    shutil.rmtree(gen_dir / "__pycache__", ignore_errors=True)  # honest: no stale cache
    (gen_dir / "staged_gen_kernel.py").write_text(
        build_src(bg_rules, "first_match", seed=202, fname="kern"))
    child_py = gen_dir / "_child.py"
    child_py.write_text(CHILD.format(gen=str(gen_dir), here=str(HERE), rules=bg_rules))

    serve_args = make_args(ROWS, "first_match", serve_rules)
    k, _ = compile_kernel(build_src(serve_rules, "first_match", seed=101, fname="serve"),
                          "serve", serve_args, nogil=True)
    for _ in range(5):
        k(*serve_args)
    base = serve_until(k, serve_args, threading.Event(), baseline_s)

    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, str(child_py)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    during = []
    while proc.poll() is None:
        t1 = time.perf_counter()
        k(*serve_args)
        during.append((time.perf_counter() - t1) * 1e3)
    out_txt, err_txt = proc.communicate()
    wall = time.perf_counter() - t0

    res = {
        "child_wall_s": wall,
        "child_rc": proc.returncode,
        "child_stdout": out_txt.strip(),
        "child_stderr": err_txt.strip()[-500:],
        "serving_baseline_median_ms": med(base),
        "serving_during_median_ms": med(during),
        "serving_during_p95_ms": pct(during, 95),
        "serving_during_max_ms": max(during) if during else None,
        "serving_latency_inflation": med(during) / med(base),
        "serving_calls": len(during),
    }

    sys.path.insert(0, str(gen_dir))
    import importlib
    m = importlib.import_module("staged_gen_kernel")
    args = make_args(ROWS, "first_match", bg_rules)
    d = njit(cache=True)(m.kern)
    t0 = time.perf_counter()
    d(*args)
    res["parent_cache_load_s"] = time.perf_counter() - t0
    res["parent_stats"] = str(d.stats)
    res["parent_checksum"] = int(args[-1].sum())
    return res


# -------------------------------------------------------------------- phase: memory
def phase_memory(n_gens=3, n_rules=30, retain=True):
    res = {"n_rules_per_generation": n_rules, "retain": retain, "steps": []}
    args = make_args(ROWS, "first_match", n_rules)
    gc.collect()
    base = rss_mb()
    res["rss_after_imports_and_data_mb"] = base
    held = []
    prev = base
    for g in range(n_gens):
        src = build_src(n_rules, "first_match", seed=900 + g, fname=f"gen{g}")
        disp, csec = compile_kernel(src, f"gen{g}", args)
        if retain:
            held.append(disp)
        else:
            del disp  # a staged generation that was never activated, then dropped
        gc.collect()
        now = rss_mb()
        res["steps"].append(
            {
                "generations_resident": g + 1,
                "emitted_lines": emit.count_lines(src),
                "compile_s": csec,
                "rss_mb": now,
                "delta_vs_previous_mb": now - prev,
                "delta_vs_baseline_mb": now - base,
            }
        )
        prev = now
    # drop back to one generation
    while len(held) > 1:
        held.pop()
    gc.collect()
    time.sleep(0.2)
    res["rss_after_dropping_to_1_generation_mb"] = rss_mb()
    res["rss_reclaimed_mb"] = prev - rss_mb()
    return res


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "interference", "gilprobe", "lockstep", "swap",
                             "subproc", "memory"])
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--memory-mode", default="retain", choices=["retain", "discard"])
    a = ap.parse_args()

    env = {
        "python": sys.version.split()[0],
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: None)(),
        "switchinterval_s": sys.getswitchinterval(),
        "cpus": os.cpu_count(),
        "rows": ROWS,
    }
    import numba

    env["numba"] = numba.__version__
    env["numpy"] = np.__version__

    t_start = time.perf_counter()
    out = {"env": env}
    if a.phase in ("all", "interference"):
        out["interference"] = phase_interference(reps=a.reps)
    if a.phase in ("all", "gilprobe"):
        out["gilprobe"] = phase_gilprobe()
    if a.phase in ("all", "lockstep"):
        out["lockstep"] = phase_lockstep()
    if a.phase in ("all", "subproc"):
        out["subproc"] = phase_subproc()
    if a.phase in ("all", "swap"):
        out["swap"] = phase_swap()
    if a.phase == "memory":
        out["memory"] = phase_memory(retain=(a.memory_mode == "retain"))
    elif a.phase == "all":
        # clean processes: RSS deltas must not include the threading phases' garbage
        out["memory"] = {}
        for mode in ("retain", "discard"):
            f = HERE / f"_memory_{mode}.json"
            p = subprocess.run([sys.executable, str(HERE / "run.py"), "--phase", "memory",
                                "--memory-mode", mode, "--out", str(f)],
                               capture_output=True, text=True)
            out["memory"][mode] = (json.loads(f.read_text())["memory"] if p.returncode == 0
                                   else {"error": p.stderr[-2000:]})
    out["wall_s"] = time.perf_counter() - t_start
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
