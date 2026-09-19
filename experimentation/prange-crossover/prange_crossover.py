"""
E-F — the prange crossover (doc 06 O6 / doc 01 §4 / doc 05 §5.1 / doc 08 §4, §8).

Measures, for a row loop whose body cost is an explicit axis:

  1. serial vs prange wall-clock across a row sweep  -> where is the crossover?
  2. whether "a light body never wins" and "~5k / 7.8x at 5M" hold.
  3. how long a warmup measurement of both variants actually costs.
  4. whether the variant choice is reproducible run to run (doc 08 §8 wants it
     in the audit record, which is only meaningful if it is stable).
  5. parallel=True compile cost vs parallel=False, per shape.

Everything reported is measured in this process. Nothing is estimated.

Run:
    /path/to/.venv/bin/python prange_crossover.py            # full run
    ... --quick                                              # small sweep
    ... --budget 900                                         # seconds, soft cap

Writes results.json next to itself.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import numba
from numba import njit, prange  # noqa: F401  (referenced by generated source)

HERE = Path(__file__).resolve().parent
T0 = time.perf_counter()


def elapsed() -> float:
    return time.perf_counter() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# Bodies.  Each is the *inside* of the row loop, indented 8 spaces, writing
# out[i] from x[i], y[i].  Body cost is the axis doc 01 §4 says sets the
# crossover, so it is made explicit here.
# --------------------------------------------------------------------------

BODY_TRIVIAL = """\
        out[i] = x[i] + y[i] * 1.5
"""

# ~12 flops, straight-line, no loop.  ("~10 flops" in the task text.)
BODY_MEDIUM = """\
        a = x[i]
        b = y[i]
        c = a * 1.000001 + b
        d = c * 0.5 - a * 0.25
        e = d * d + c * b
        out[i] = e * 0.125 + d * 1.5 - c
"""

# Serial dependence chain inside the inner loop: LLVM may unroll it but cannot
# vectorise it away and (fastmath=False) cannot reassociate it into a closed
# form.  So K really is K units of per-row work.
BODY_INNER = """\
        acc = x[i]
        b = y[i]
        for k in range({K}):
            acc = acc * 1.0000001 + b * 0.5 - acc * 0.25
        out[i] = acc
"""

BODIES = {
    "trivial": BODY_TRIVIAL,
    "medium10": BODY_MEDIUM,
    "inner64": BODY_INNER.format(K=64),
    "inner256": BODY_INNER.format(K=256),
}

KERNEL_TEMPLATE = """\
import numpy as np
from numba import njit, prange

@njit(cache=False, parallel={parallel}, fastmath=False)
def {name}(x, y, out):
    n = x.shape[0]
    for i in {loop}(n):
{body}
    return out
"""


def build_kernel(name: str, body: str, parallel: bool):
    """Compile-on-first-call kernel from generated source. Returns (fn, source)."""
    src = KERNEL_TEMPLATE.format(
        name=name,
        parallel="True" if parallel else "False",
        loop="prange" if parallel else "range",
        body=body,
    )
    ns: dict = {}
    exec(compile(src, f"<gen:{name}>", "exec"), ns)
    return ns[name], src


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------


def time_once(fn, x, y, out) -> float:
    t = time.perf_counter()
    fn(x, y, out)
    return time.perf_counter() - t


def time_reps(fn, x, y, out, reps: int) -> list[float]:
    return [time_once(fn, x, y, out) for _ in range(reps)]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


# --------------------------------------------------------------------------
# Phase 1+4 — the sweep, run as T independent trials so the variant choice
# can be checked for reproducibility (doc 08 §8).
# --------------------------------------------------------------------------


def run_sweep(row_grid, trials, budget, results):
    """For each (body, rows): compile serial+prange once, then time both
    variants `trials` times with adaptive repeat counts.

    `row_grid` maps body name -> list of row counts.  The grids differ per body
    on purpose: the heavy bodies cross far below 1k so they need small points,
    the light bodies need large ones.  Every body still covers the
    1k/10k/100k/1M/5M points the brief asked for.
    """
    bodies = list(row_grid)
    kernels = {}
    compile_times = {}

    # One throwaway compile so the first *measured* compile does not pay
    # numba/LLVM first-use initialisation.
    warm, _ = build_kernel("kwarm", BODY_TRIVIAL, parallel=False)
    warm(np.ones(4), np.ones(4), np.empty(4))
    warmp, _ = build_kernel("kwarmp", BODY_TRIVIAL, parallel=True)
    warmp(np.ones(4), np.ones(4), np.empty(4))  # also starts the thread pool
    try:
        results["threading_layer"] = numba.threading_layer()
    except Exception as exc:  # pragma: no cover
        results["threading_layer"] = f"unavailable: {exc}"

    max_rows = max(max(v) for v in row_grid.values())
    rng = np.random.default_rng(0xC0FFEE)
    x_full = rng.random(max_rows) + 0.5
    y_full = rng.random(max_rows) + 0.5
    out_full = np.empty(max_rows)
    out_check = np.empty(max_rows)

    for bname in bodies:
        for parallel in (False, True):
            kname = f"k_{bname}_{'par' if parallel else 'ser'}"
            fn, _src = build_kernel(kname, BODIES[bname], parallel)
            t = time.perf_counter()
            fn(x_full[:1024], y_full[:1024], out_full[:1024])  # forces compile
            compile_times[(bname, parallel)] = time.perf_counter() - t
            kernels[(bname, parallel)] = fn
            log(f"compiled {kname}: {compile_times[(bname, parallel)] * 1e3:8.1f} ms")

    results["sweep_compile_ms"] = {
        f"{b}/{'prange' if p else 'serial'}": v * 1e3 for (b, p), v in compile_times.items()
    }

    cells = []
    for bname in bodies:
        for n in row_grid[bname]:
            if elapsed() > budget:
                cells.append(
                    {"body": bname, "rows": n, "skipped": "global time budget exhausted"}
                )
                log(f"SKIP {bname}/{n} — budget")
                continue

            x, y = x_full[:n], y_full[:n]
            out = out_full[:n]
            ser = kernels[(bname, False)]
            par = kernels[(bname, True)]

            # correctness: prange must not change the answer
            ser(x, y, out_check[:n])
            par(x, y, out)
            bitwise_identical = bool(np.array_equal(out_check[:n], out, equal_nan=True))

            # one untimed pass each to settle caches
            ser(x, y, out)
            par(x, y, out)

            t_est = min(time_once(ser, x, y, out), time_once(par, x, y, out))
            t_big = max(time_once(ser, x, y, out), time_once(par, x, y, out))
            # adaptive: ~0.25 s of timing per (variant, trial), 3..15 reps
            reps = int(max(3, min(15, 0.25 / max(t_big, 1e-9))))
            n_trials = trials if t_big * reps * trials * 2 < 12.0 else 3
            del t_est

            trial_rows = []
            for _ in range(n_trials):
                s = time_reps(ser, x, y, out, reps)
                p = time_reps(par, x, y, out, reps)
                trial_rows.append(
                    {
                        "serial_med": med(s),
                        "prange_med": med(p),
                        "serial_min": min(s),
                        "prange_min": min(p),
                        "serial_all": s,
                        "prange_all": p,
                    }
                )

            ser_meds = [t["serial_med"] for t in trial_rows]
            par_meds = [t["prange_med"] for t in trial_rows]
            speedups = [sm / pm for sm, pm in zip(ser_meds, par_meds)]
            picks = ["prange" if sp > 1.0 else "serial" for sp in speedups]

            cell = {
                "body": bname,
                "rows": n,
                "reps": reps,
                "trials": n_trials,
                "bitwise_identical": bitwise_identical,
                "serial_med_s": med(ser_meds),
                "prange_med_s": med(par_meds),
                "serial_ns_per_row": med(ser_meds) * 1e9 / n,
                "prange_ns_per_row": med(par_meds) * 1e9 / n,
                "speedup_med": med(speedups),
                "speedup_min": min(speedups),
                "speedup_max": max(speedups),
                "picks": picks,
                "pick_unanimous": len(set(picks)) == 1,
                "trial_detail": trial_rows,
            }
            cells.append(cell)
            log(
                f"{bname:9s} n={n:>9,d} ser={med(ser_meds)*1e3:9.3f}ms "
                f"par={med(par_meds)*1e3:9.3f}ms speedup={med(speedups):6.3f}x "
                f"[{min(speedups):.3f}-{max(speedups):.3f}] picks={'/'.join(sorted(set(picks)))}"
                f"{'' if bitwise_identical else '  !!NOT BITWISE IDENTICAL!!'}"
            )
    results["sweep"] = cells
    return kernels


# --------------------------------------------------------------------------
# Phase 2 — what a warmup measurement actually costs, and whether a *cheap*
# probe transfers to the production row count.
# --------------------------------------------------------------------------


def run_warmup_probe(kernels, bodies, probe_rows, target_rows, budget, results):
    rng = np.random.default_rng(7)
    max_rows = max(max(probe_rows), max(target_rows))
    x = rng.random(max_rows) + 0.5
    y = rng.random(max_rows) + 0.5
    out = np.empty(max_rows)

    rows_out = []
    for bname in bodies:
        ser, par = kernels[(bname, False)], kernels[(bname, True)]
        for n in probe_rows:
            if elapsed() > budget:
                rows_out.append({"body": bname, "probe_rows": n, "skipped": "budget"})
                continue
            for k in (1, 3, 5):
                # exactly what a warmup measurer would do: k reps of each
                # variant on a probe batch, median, pick the faster.
                t0 = time.perf_counter()
                s = time_reps(ser, x[:n], y[:n], out[:n], k)
                p = time_reps(par, x[:n], y[:n], out[:n], k)
                cost = time.perf_counter() - t0
                rows_out.append(
                    {
                        "body": bname,
                        "probe_rows": n,
                        "k": k,
                        "probe_cost_s": cost,
                        "pick": "prange" if med(s) > med(p) else "serial",
                        "ratio": med(s) / med(p),
                    }
                )
    results["warmup_probe"] = rows_out
    for r in rows_out:
        if "skipped" in r:
            continue
        log(
            f"probe {r['body']:9s} n={r['probe_rows']:>8,d} k={r['k']} "
            f"cost={r['probe_cost_s']*1e3:8.2f}ms pick={r['pick']:6s} ratio={r['ratio']:.3f}"
        )


# --------------------------------------------------------------------------
# Phase 3 — flip rate of a *cheap* warmup pick, repeated many times.
# This is the doc 08 §8 question: is the recorded variant reproducible?
# --------------------------------------------------------------------------


def choose_flip_cases(results, max_serial_s: float):
    """Per body: the 3 cells with measured speedup nearest 1.0, plus the
    cheapest and a mid control — all restricted to cells cheap enough to
    re-measure 15 times."""
    cases: list[tuple[str, int]] = []
    by_body: dict[str, list] = {}
    for c in results["sweep"]:
        if "skipped" in c or c["serial_med_s"] > max_serial_s:
            continue
        by_body.setdefault(c["body"], []).append(c)
    for body, cells in by_body.items():
        cells_sorted = sorted(cells, key=lambda c: abs(np.log(max(c["speedup_med"], 1e-9))))
        chosen = [c["rows"] for c in cells_sorted[:3]]
        chosen.append(min(c["rows"] for c in cells))
        chosen.append(max(c["rows"] for c in cells))
        for r in dict.fromkeys(chosen):
            cases.append((body, r))
    return cases


def run_flip_test(kernels, cases, n_picks, k, budget, results):
    if not cases:
        results["flip_test"] = []
        log("flip test: no cells cheap enough — skipped")
        return
    rng = np.random.default_rng(11)
    max_rows = max(n for _, n in cases)
    x = rng.random(max_rows) + 0.5
    y = rng.random(max_rows) + 0.5
    out = np.empty(max_rows)

    rows_out = []
    for bname, n in cases:
        if elapsed() > budget:
            rows_out.append({"body": bname, "rows": n, "skipped": "budget"})
            continue
        ser, par = kernels[(bname, False)], kernels[(bname, True)]
        picks, ratios = [], []
        for _ in range(n_picks):
            s = time_reps(ser, x[:n], y[:n], out[:n], k)
            p = time_reps(par, x[:n], y[:n], out[:n], k)
            r = med(s) / med(p)
            ratios.append(r)
            picks.append("prange" if r > 1.0 else "serial")
        frac_par = picks.count("prange") / len(picks)
        rows_out.append(
            {
                "body": bname,
                "rows": n,
                "k": k,
                "n_picks": n_picks,
                "frac_prange": frac_par,
                "ratio_med": med(ratios),
                "ratio_min": min(ratios),
                "ratio_max": max(ratios),
                "ratio_spread_pct": (max(ratios) - min(ratios)) / med(ratios) * 100.0,
            }
        )
        log(
            f"flip  {bname:9s} n={n:>9,d} frac_prange={frac_par:5.2f} "
            f"ratio med={med(ratios):.3f} [{min(ratios):.3f}-{max(ratios):.3f}] "
            f"spread={(max(ratios)-min(ratios))/med(ratios)*100:5.1f}%"
        )
    results["flip_test"] = rows_out


# --------------------------------------------------------------------------
# Phase 5 — parallel=True compile cost, per shape.
# --------------------------------------------------------------------------


def branchy_body(nbranch: int) -> str:
    """nbranch flat if/else groups, as in doc 01 §4b's 'flat 50/100 branches'."""
    lines = ["        v = x[i]", "        w = y[i]"]
    for b in range(nbranch):
        thr = 0.1 + 0.007 * b
        lines.append(f"        if w > {thr:.6f}:")
        lines.append(f"            v = v * {1.0 + 0.001 * b:.6f} + {0.5 + b * 0.01:.4f}")
        lines.append("        else:")
        lines.append(f"            v = v - {0.25 + b * 0.005:.4f} * w")
    lines.append("        out[i] = v")
    return "\n".join(lines) + "\n"


def run_compile_cost(shapes, budget, results):
    rows_out = []
    xs = np.ones(1024)
    ys = np.ones(1024)
    os_ = np.empty(1024)
    for label, body in shapes:
        if elapsed() > budget:
            rows_out.append({"shape": label, "skipped": "global time budget exhausted"})
            log(f"SKIP compile-cost {label} — budget")
            continue
        times = {}
        for parallel in (False, True):
            name = f"cc_{label}_{'par' if parallel else 'ser'}"
            fn, _ = build_kernel(name, body, parallel)
            t = time.perf_counter()
            fn(xs, ys, os_)
            times[parallel] = time.perf_counter() - t
        rows_out.append(
            {
                "shape": label,
                "serial_compile_s": times[False],
                "parallel_compile_s": times[True],
                "ratio": times[True] / times[False],
            }
        )
        log(
            f"compile {label:14s} serial={times[False]*1e3:9.1f}ms "
            f"parallel={times[True]*1e3:9.1f}ms ratio={times[True]/times[False]:5.2f}x"
        )
    results["compile_cost"] = rows_out


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--budget", type=float, default=760.0, help="soft wall-clock cap (s)")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    results = {
        "meta": {
            "python": sys.version.split()[0],
            "numba": numba.__version__,
            "llvmlite": __import__("llvmlite").__version__,
            "numpy": np.__version__,
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "numba_num_threads": numba.config.NUMBA_NUM_THREADS,
            "threading_layer_cfg": str(numba.config.THREADING_LAYER),
            "argv": sys.argv[1:],
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    log(f"meta: {json.dumps(results['meta'])}")

    bodies = list(BODIES)
    # Required points from the brief, present for every body:
    REQUIRED = [1_000, 10_000, 100_000, 1_000_000, 5_000_000]
    # Light bodies: extra resolution at the top, where their crossover (if any)
    # would have to be.  Heavy bodies: extra resolution at the bottom, because
    # the smoke run showed them already winning at 1k.
    LIGHT = sorted(set(REQUIRED + [2_000, 5_000, 20_000, 50_000, 200_000, 500_000, 2_000_000]))
    HEAVY = sorted(set(REQUIRED + [100, 200, 500, 2_000, 5_000, 20_000, 50_000]))
    if args.quick:
        row_grid = {b: [1_000, 10_000, 100_000] for b in bodies}
        trials = 3
    else:
        row_grid = {
            "trivial": LIGHT,
            "medium10": LIGHT,
            "inner64": HEAVY,
            "inner256": HEAVY,
        }
        trials = args.trials
    results["row_grid"] = row_grid

    budget = args.budget

    log("=== phase 1: serial vs prange sweep ===")
    kernels = run_sweep(row_grid, trials, budget * 0.55, results)

    log("=== phase 2: warmup probe cost ===")
    run_warmup_probe(
        kernels,
        bodies,
        probe_rows=[1_000, 10_000, 100_000] if not args.quick else [1_000, 10_000],
        target_rows=[1_000_000],
        budget=budget * 0.72,
        results=results,
    )

    log("=== phase 3: pick reproducibility (flip rate) ===")
    # Pick the cases from phase 1: per body, the rows whose measured speedup is
    # closest to 1.0 (where a warmup measurer is most likely to flip), plus one
    # clearly-serial and one clearly-prange control.  Bounded by serial cost so
    # 15 repeated picks stay cheap.
    flip_cases = choose_flip_cases(results, max_serial_s=0.04)
    log(f"flip cases (chosen from phase 1): {flip_cases}")
    run_flip_test(kernels, flip_cases, n_picks=15, k=3, budget=budget * 0.85, results=results)

    log("=== phase 5: parallel=True compile cost ===")
    shapes = [("trivial", BODY_TRIVIAL), ("medium10", BODY_MEDIUM)]
    shapes += [(f"inner{k}", BODY_INNER.format(K=k)) for k in (64, 256)]
    shapes += [(f"branch{b}", branchy_body(b)) for b in (10, 25, 50, 100)]
    if args.quick:
        shapes = shapes[:5]
    run_compile_cost(shapes, budget * 0.99, results)

    results["meta"]["total_s"] = elapsed()
    Path(args.out).write_text(json.dumps(results, indent=1))
    log(f"wrote {args.out}  (total {elapsed():.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
